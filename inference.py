import os

import torch

from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
from llama.quantize import QuantizedLinear, quantize_model

def inference():
    """Smoke test: load the dense FP16 model and print a few sample generations."""
    torch.manual_seed(1)

    checkpoint_dir = "/project2/saifhash_1190/llama/checkpoints/Llama3.2-1B"
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    model_path = os.path.join(checkpoint_dir, "consolidated.00.pth")

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor) # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=True)
    device = "cuda"
    model = quantize_model(model=model, group_size=128)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    buffer_elements = sum(b.numel() for b in model.buffers())
    param_size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffer_size_bytes = sum(b.element_size() * b.numel() for b in model.buffers()) # scales and zero-point size
    total_size_mb = (param_size_bytes + buffer_size_bytes) / (1024 * 1024)

    print(f"Total Parameters: {total_params:,}")
    print(f"Total elements in buffers: {buffer_elements:,}")
    print(f"Model Size on Disk/Memory: {total_size_mb:.2f} MB")
    print("==================================\n")

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9, kv_caching=True, device=device)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    inference()
