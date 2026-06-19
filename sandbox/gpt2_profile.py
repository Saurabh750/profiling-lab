import argparse
import contextlib
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel


MODEL_CONFIGS = {
    "gpt2":        "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large":  "gpt2-large",
    "gpt2-xl":     "gpt2-xl",
}


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    choices=list(MODEL_CONFIGS.keys()), default="gpt2")
    p.add_argument("--batch",    type=int, default=1)
    p.add_argument("--seq_len",  type=int, default=512)
    p.add_argument("--dtype",    choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile",  action="store_true")
    p.add_argument("--warmup",   action="store_true")
    p.add_argument("--trace_dir", default="./traces/gpt2")
    return p.parse_args()


def build_model(model_name: str, dtype: torch.dtype) -> torch.nn.Module:
    config = GPT2Config.from_pretrained(MODEL_CONFIGS[model_name])
    model = GPT2LMHeadModel(config).to(device="cuda", dtype=dtype).eval()
    return model


def attach_profiler_hooks(model: torch.nn.Module) -> list:
    """
    Wraps each named submodule in a record_function region so the chrome
    trace shows per-component timing (e.g. transformer.h.0.attn vs .mlp).
    Returns the hook handles so they can be removed later if needed.
    """
    handles = []

    @contextlib.contextmanager
    def record(name):
        with torch.profiler.record_function(name):
            yield

    class RecordFunctionHook:
        def __init__(self, name):
            self.name = name
            self._ctx = None

        def pre(self, module, inputs):
            self._ctx = torch.profiler.record_function(self.name)
            self._ctx.__enter__()

        def post(self, module, inputs, output):
            if self._ctx is not None:
                self._ctx.__exit__(None, None, None)
                self._ctx = None

    for name, module in model.named_modules():
        if name == "":
            continue
        hook = RecordFunctionHook(name)
        handles.append(module.register_forward_pre_hook(hook.pre))
        handles.append(module.register_forward_hook(hook.post))

    return handles


def main():
    args = parse_arguments()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = build_model(args.model, dtype)

    if args.compile:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    attach_profiler_hooks(model)

    input_ids = torch.randint(
        0, 50257, (args.batch, args.seq_len), device="cuda"
    )

    def step():
        with torch.profiler.record_function("gpt2_forward"):
            with torch.no_grad():
                return model(input_ids)

    if args.warmup:
        for _ in range(3):
            step()
        torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)
    compile_tag = "compile" if args.compile else "eager"
    warmup_tag  = "warm" if args.warmup else "cold"
    tag = f"{args.model}_b{args.batch}_s{args.seq_len}_{args.dtype}_{warmup_tag}_{compile_tag}"

    table_path = os.path.join(args.trace_dir, f"{tag}.txt")
    trace_path = os.path.join(args.trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()

    torch.cuda.synchronize()

    print(f"saving trace  ... {trace_path}")
    prof.export_chrome_trace(trace_path)

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print(f"saving table  ... {table_path}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
