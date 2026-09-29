"""
Mixture-of-Experts (MoE) FeedForward layer for Llama 3.2-1B

This module implements two MoE variants:

1. ``slice`` init — split the pretrained dense FFN into N smaller expert FFNs,
   each with hidden_dim / N.  A learned router selects top-K experts per token.

2. ``lora`` init — keep the original dense FFN **frozen** and attach N lightweight
   LoRA adapters as "experts".  The router selects which adapters to activate.
   This is far more parameter-efficient than copying the full FFN.

Students: Complete the classes and functions marked with TODO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================================================================
# Part A: Slice-based MoE
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

        # TODO Step 1: Compute router scores and select top-K experts
        # - Pass x_flat through self.router to get logits (num_tokens, num_experts)
        router_logits = self.router(x_flat.to(self.router.weight.dtype))
        # - Apply softmax to get routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)
        # - Use torch.topk to pick the top-K experts and their weights
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1) # indices
        # - Renormalize the top-K weights so they sum to 1 per token
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9) # weights are probabilities
        # - Save top_k_indices to self._last_routing_indices (detached)
        top_k_weights = top_k_weights.to(x_flat.dtype)
        self._last_routing_indices = top_k_indices.detach()

        # TODO Step 2: Compute the weighted sum of selected expert outputs
        # - Think about what the output shape should be
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            token_pos, slot_pos = torch.where(top_k_indices == expert_idx) # rows and cols
            if token_pos.numel() == 0:
                continue
            expert_input = x_flat[token_pos]
            # - For each token, for each of its top-K experts:
            #     call self._expert_forward(expert_idx, token_input)
            expert_output = self._expert_forward(expert_idx, expert_input)
            #     accumulate: weight * expert_output
            weights = top_k_weights[token_pos, slot_pos].unsqueeze(-1) # (n, 1)
            output.index_add_(0, token_pos, weights * expert_output) # weighted avg

        # - Hint: _expert_forward is provided below

        # TODO Step 3: Reshape back to original shape and return
        return output.reshape(original_shape)

        # raise NotImplementedError("Complete the MoEFeedForward forward method")

    def _expert_forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Run a single expert's SwiGLU FFN."""
        return self.expert_w2[expert_idx](
            F.silu(self.expert_w1[expert_idx](x)) * self.expert_w3[expert_idx](x)
        )


# ===================================================================
# Part B: LoRA-based MoE
# ===================================================================
#
# Motivation: slice mode (Part A) splits the pretrained FFN into N smaller
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
# Q: How many parameters does each LoRA expert add compared to one full expert?
# A: Total of 2,228,224/4 = 557,056 # of parameters are added by each LoRA expert.

# Q: With the base FFN frozen, what guarantees the pre-training quality is
#    preserved even before fine-tuning?
# A: Because that entire MoE/LoRA addition equals exactly zero before any backward passes
#    or weight updates happen, the output logits of the new LoRAMoEFeedForward layer are
#    mathematically identical to the outputs of the original dense model, So the pre-training quality
#    is preserved.

class LoRAExpert(nn.Module):
    """
    A single LoRA adapter used as a lightweight expert.

    Architecture:  x → A (down-project to rank r) → B (up-project back to dim)

    TODO: Implement __init__ and forward.

    Hints:
    - You need two linear layers without bias
    - Think about which matrix should be initialized to zero, and why
      (What should the adapter's output be before any training happens?)
    - The base model runs in FP16, but LoRA gradients in FP16 will NaN.
      Use dtype=torch.float32 for LoRA layers to keep training stable.
    - In forward, cast input to float32, compute, then cast output back.
    - Standard LoRA scales output by alpha/rank so that changing rank
      doesn't change the magnitude. Store this as self.scaling.
    """

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # TODO: Store the scaling factor alpha / rank
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # TODO: Create lora_A (dim → rank) and lora_B (rank → dim)
        #       Use dtype=torch.float32 for both (training stability)
        self.lora_A = nn.Linear(dim, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, dim, bias=False, dtype=torch.float32)
        # TODO: Initialize one of them to zero — which one ensures
        #       the adapter outputs zero before training?
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # raise NotImplementedError("Complete the LoRAExpert __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Cast x to float32, pass through A then B,
        #       multiply by self.scaling, cast back to x.dtype
        out = self.lora_B(self.lora_A(x.to(torch.float32))) * self.scaling
        return out.to(x.dtype)
        # raise NotImplementedError("Complete the LoRAExpert forward")


class LoRAMoEFeedForward(nn.Module):
    """
    Frozen shared FFN + routed LoRA experts.

    The base FFN is the original pretrained FeedForward — kept frozen so it
    adds zero extra memory.  Each LoRA expert adds only 2 * dim * rank
    parameters, which is tiny compared to a full FFN copy.

    TODO: Implement __init__ and forward.

    Hints for __init__:
    - Store the original FeedForward as self.base_ff and freeze its parameters
    - Create a router (use dtype=torch.float32 like the LoRA layers)
    - Create N LoRAExpert instances in a ModuleList
    - Don't forget self._last_routing_indices for analysis

    Hints for forward:
    - The forward combines two things:
        (a) the frozen base FFN output (no gradient needed)
        (b) the weighted sum of top-K LoRA expert outputs (has gradient)
    - Route in float32: cast x_flat to float32 before the router
    - The routing logic is the same as MoEFeedForward
    - Final output = base_output + lora_output
    """

    def __init__(self, base_ff: nn.Module, dim: int, num_experts: int = 4,
                 top_k: int = 2, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank

        # TODO: Store and freeze the base FFN
        self.base_ff = base_ff
        for p in self.base_ff.parameters():
            p.requires_grad = False
        # TODO: Create the router
        self.router = nn.Linear(dim, num_experts, bias=False, dtype=torch.float32)
        # TODO: Create N LoRA experts
        self.lora_experts = nn.ModuleList([
            LoRAExpert(dim, rank=rank, alpha=alpha) for _ in range(num_experts)
        ])
        self._last_routing_indices = None

        # raise NotImplementedError("Complete the LoRAMoEFeedForward __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass
        # 1. Flatten x to (num_tokens, dim)
        x_flat = x.reshape(-1, self.dim)
        # 2. Compute base FFN output (frozen, use torch.no_grad())
        with torch.no_grad():
            base_output = self.base_ff(x)
        x_flat_fp32 = x_flat.to(torch.float32)
        # 3. Route tokens to top-K LoRA experts (same as MoEFeedForward)
        router_logits = self.router(x_flat_fp32)
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        # 4. Compute weighted sum of LoRA expert outputs
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
        # 5. Return base_output + lora_output, reshaped to original shape
        lora_output = lora_output.to(x.dtype).reshape(x.shape)
        return base_output + lora_output

        # raise NotImplementedError("Complete the LoRAMoEFeedForward forward")


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
            # TODO: Create a LoRAMoEFeedForward that wraps old_ff
            # - Pass old_ff as the base_ff argument
            new_ff = LoRAMoEFeedForward(
                base_ff=old_ff,
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            # - Move to correct device (but NOT dtype — LoRA params must stay float32)
            new_ff.router = new_ff.router.to(device)
            for exp in new_ff.lora_experts:
                exp.to(device)

            # - Replace layer.feed_forward
            layer.feed_forward = new_ff
            # - Collect trainable params: router + all LoRA expert params
            trainable_params.append(new_ff.router.weight)
            for exp in new_ff.lora_experts:
                trainable_params.append(exp.lora_A.weight)
                trainable_params.append(exp.lora_B.weight)

            # raise NotImpleme/ntedError("Complete LoRA MoE conversion")

        elif init_mode == "slice":
            # TODO: Create a MoEFeedForward and initialize expert weights
            new_ff = MoEFeedForward(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
            ).to(device=device, dtype=dtype)
            # by slicing the original FFN.
            with torch.no_grad():
                for i in range(num_experts):
                    slice_start = i * (hidden_dim // num_experts)
                    slice_end = (i + 1) * (hidden_dim // num_experts)
                #   expert_w1[i] ← old_ff.w1.weight[slice_start:slice_end, :]
                    new_ff.expert_w1[i].weight.copy_(old_ff.w1.weight[slice_start:slice_end, :])
                #   expert_w2[i] ← old_ff.w2.weight[:, slice_start:slice_end]
                    new_ff.expert_w2[i].weight.copy_(old_ff.w2.weight[:, slice_start:slice_end])
                #   expert_w3[i] ← old_ff.w3.weight[slice_start:slice_end, :]
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

            # raise NotImplementedError("Complete slice MoE conversion")

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
            # Llama tiktoken tokenizer requires bos/eos kwargs.
            # Try that first; fall back to HF-style on TypeError.
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

    # raise NotImplementedError("Complete the expert load stats collection")
