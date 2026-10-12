"""
Mixture-of-Experts (MoE) FeedForward layer for Llama 3.2-1B

This module implements two MoE variants:

1. ``slice`` init — split the pretrained dense FFN into N smaller expert FFNs,
   each with hidden_dim / N.  A learned router selects top-K experts per token.

2. ``lora`` init — keep the original dense FFN **frozen** and attach N lightweight
   LoRA adapters as "experts".  The router selects which adapters to activate.
   This is far more parameter-efficient than copying the full FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================================================================
# Slice-based MoE
# ===================================================================

class MoEFeedForward(nn.Module):
    """
    Mixture-of-Experts FeedForward layer (slice variant).

    Each expert has the same SwiGLU structure (w1, w2, w3) as the original FFN,
    but with hidden_dim / num_experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        """
        Args:
            dim: input/output dimension (model embedding dim)
            hidden_dim: hidden dimension of the original dense FFN
            num_experts: number of experts (N)
            top_k: number of experts activated per token (K)
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = hidden_dim // num_experts

        # Router: maps input to expert scores
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Experts: each is a small SwiGLU FFN (same structure as Llama FeedForward)
        self.expert_w1 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])
        self.expert_w2 = nn.ModuleList([nn.Linear(self.expert_hidden_dim, dim, bias=False) for _ in range(num_experts)])
        self.expert_w3 = nn.ModuleList([nn.Linear(dim, self.expert_hidden_dim, bias=False) for _ in range(num_experts)])

        # For tracking expert activations (used in analysis)
        self._last_routing_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, dim)

        Returns:
            output tensor of shape (batch_size, seq_len, dim)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]

        # Select and renormalize the top-k routing probabilities per token.
        router_logits = self.router(x_flat.to(self.router.weight.dtype))
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1) # indices
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9) # weights are probabilities
        top_k_weights = top_k_weights.to(x_flat.dtype)
        self._last_routing_indices = top_k_indices.detach()

        # Dispatch selected tokens to each expert and accumulate their outputs.
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            token_pos, slot_pos = torch.where(top_k_indices == expert_idx) # rows and cols
            if token_pos.numel() == 0:
                continue
            expert_input = x_flat[token_pos]
            expert_output = self._expert_forward(expert_idx, expert_input)
            weights = top_k_weights[token_pos, slot_pos].unsqueeze(-1) # (n, 1)
            output.index_add_(0, token_pos, weights * expert_output) # weighted avg

        return output.reshape(original_shape)

    def _expert_forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run a single expert's SwiGLU FFN."""
        return self.expert_w2[expert_idx](
            F.silu(self.expert_w1[expert_idx](x)) * self.expert_w3[expert_idx](x)
        )


# ===================================================================
# LoRA-based MoE
# ===================================================================
#
# Motivation: slice mode splits the pretrained FFN into N smaller
# experts. With top_k < N active experts per token and only a few hundred
# fine-tuning samples, each expert loses ~half of its original capacity and
# the router hasn't learned to specialize — quality drops sharply. LoRA MoE
# solves this by keeping the full FFN intact and adding tiny "diff" experts
# on top.
#
# Recall from LoRA (Hu et al., 2021):
#   ΔW = B @ A,  where A ∈ R^{r×d}, B ∈ R^{d×r}, r << d
#
# Here each expert is one such (A, B) pair added to the frozen FFN output:
#   output = base_ffn(x) + Σ_k  weight_k · lora_expert_k(x)
#
# At initialization, B is zero → every expert contributes zero →
# the converted model behaves identically to the original dense model.
#
# Zero-initializing B makes every adapter output zero initially, so conversion
# preserves the pretrained model's behavior before fine-tuning.

class LoRAExpert(nn.Module):
    """
    A single LoRA adapter used as a lightweight expert.

    Architecture:  x → A (down-project to rank r) → B (up-project back to dim)

    Parameters and intermediate adapter activations use float32 for stable
    fine-tuning, while outputs are cast back to the input dtype.
    """

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep adapter parameters in float32 for training stability.
        self.lora_A = nn.Linear(dim, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, dim, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora_B(self.lora_A(x.to(torch.float32))) * self.scaling
        return out.to(x.dtype)


class LoRAMoEFeedForward(nn.Module):
    """
    Frozen shared FFN + routed LoRA experts.

    The base FFN is the original pretrained FeedForward — kept frozen so it
    adds zero extra memory.  Each LoRA expert adds only 2 * dim * rank
    parameters, which is tiny compared to a full FFN copy.

    The router and adapters use float32; the frozen base FFN retains its
    original dtype.
    """

    def __init__(self, base_ff: nn.Module, dim: int, num_experts: int = 4,
                 top_k: int = 2, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank

        self.base_ff = base_ff
        for p in self.base_ff.parameters():
            p.requires_grad = False
        self.router = nn.Linear(dim, num_experts, bias=False, dtype=torch.float32)
        self.lora_experts = nn.ModuleList([
            LoRAExpert(dim, rank=rank, alpha=alpha) for _ in range(num_experts)
        ])
        self._last_routing_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self.dim)
        with torch.no_grad():
            base_output = self.base_ff(x)
        x_flat_fp32 = x_flat.to(torch.float32)
        router_logits = self.router(x_flat_fp32)
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        self._last_routing_indices = top_k_indices.detach()
        lora_output = torch.zeros_like(x_flat_fp32)
        for expert_idx in range(self.num_experts):
            token_pos, slot_pos = torch.where(top_k_indices == expert_idx)
            if token_pos.numel() == 0:
                continue

            expert_input = x_flat[token_pos]
            expert_output = self.lora_experts[expert_idx](expert_input).to(torch.float32)
            weights = top_k_weights[token_pos, slot_pos].unsqueeze(-1)
            lora_output.index_add_(0, token_pos, weights * expert_output)
        lora_output = lora_output.to(x.dtype).reshape(x.shape)
        return base_output + lora_output


# ===================================================================
# Model conversion
# ===================================================================

def convert_to_moe(model, num_experts=4, top_k=2, init_mode="slice",
                   lora_rank=8, lora_alpha=16.0):
    """
    Replace all FeedForward layers in the model with MoE variants.

    Args:
        model: the Llama model
        num_experts: number of experts per layer
        top_k: number of active experts per token
        init_mode: ``"slice"`` or ``"lora"``
        lora_rank: rank for LoRA experts (only used when init_mode="lora")
        lora_alpha: LoRA scaling factor; effective scale = alpha / rank

    Returns:
        (model, trainable_params): the modified model and a flat list of
        parameters that should be passed to the optimizer.
    """
    trainable_params = []

    for layer in model.layers:
        old_ff = layer.feed_forward
        hidden_dim = old_ff.w1.out_features  # original hidden dim
        ref_param = old_ff.w1.weight
        device = ref_param.device
        dtype = ref_param.dtype
        dim = old_ff.w1.in_features

        if init_mode == "lora":
            new_ff = LoRAMoEFeedForward(
                base_ff=old_ff,
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            # Preserve float32 adapter parameters while moving them to the model device.
            new_ff.router = new_ff.router.to(device)
            for exp in new_ff.lora_experts:
                exp.to(device)

            layer.feed_forward = new_ff
            trainable_params.append(new_ff.router.weight)
            for exp in new_ff.lora_experts:
                trainable_params.append(exp.lora_A.weight)
                trainable_params.append(exp.lora_B.weight)

        elif init_mode == "slice":
            new_ff = MoEFeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
            ).to(device=device, dtype=dtype)
            # Initialize experts from contiguous slices of the dense FFN.
            with torch.no_grad():
                for i in range(num_experts):
                    slice_start = i * (hidden_dim // num_experts)
                    slice_end = (i + 1) * (hidden_dim // num_experts)
                    new_ff.expert_w1[i].weight.copy_(old_ff.w1.weight[slice_start:slice_end, :])
                    new_ff.expert_w2[i].weight.copy_(old_ff.w2.weight[:, slice_start:slice_end])
                    new_ff.expert_w3[i].weight.copy_(old_ff.w3.weight[slice_start:slice_end, :])

            new_ff.router = new_ff.router.to(device=device, dtype=torch.float32)
            layer.feed_forward = new_ff

            for i in range(num_experts):
                new_ff.expert_w1[i].weight.requires_grad = False
                new_ff.expert_w2[i].weight.requires_grad = False
                new_ff.expert_w3[i].weight.requires_grad = False

            # Router stays trainable (it's the only thing that learns).
            new_ff.router.weight.requires_grad = True
            trainable_params.append(new_ff.router.weight)

        else:
            raise ValueError(f"init_mode must be 'slice' or 'lora', got {init_mode!r}")

    return model, trainable_params


# ===================================================================
# Analysis helper
# ===================================================================

def get_expert_load_stats(model, tokenizer, prompts, device="cuda"):
    """
    Run prompts through the model and collect expert activation statistics.

    Returns a dict mapping layer_idx -> tensor of shape (num_experts,)
    with activation counts.
    """
    model.eval()
    stats = {}

    moe_layers = []
    for layer_idx, layer in enumerate(model.layers):
        ff = layer.feed_forward
        if isinstance(ff, (MoEFeedForward, LoRAMoEFeedForward)):
            moe_layers.append((layer_idx, ff))
            stats[layer_idx] = torch.zeros(ff.num_experts, dtype=torch.long)

    if not moe_layers:
        return stats

    with torch.no_grad():
        for prompt in prompts:
            # Prefer the Llama tiktoken API and fall back to the HF-style API.
            try:
                ids = tokenizer.encode(prompt, bos=True, eos=False)
            except TypeError:
                enc = tokenizer(prompt, return_tensors="pt")
                ids = enc["input_ids"]

            if isinstance(ids, list):
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            else:
                input_ids = ids.to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

            try:
                model(input_ids, start_pos=0)
            except TypeError:
                model(input_ids)

            for layer_idx, ff in moe_layers:
                idx = ff._last_routing_indices
                if idx is None:
                    continue
                flat = idx.reshape(-1).to("cpu")
                counts = torch.bincount(flat, minlength=ff.num_experts)
                stats[layer_idx] += counts

    return stats
