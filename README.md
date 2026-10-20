# ggufit

**Will this GGUF model actually fit on your GPU?** One command, offline, zero dependencies. It reads the model's real GGUF header — parameter count, quantization, layer layout — does the VRAM math (weights + KV cache + runtime overhead), and tells you whether your card can take it *before* you download 4 GB and meet the OOM killer.

```sh
curl -O https://raw.githubusercontent.com/techdevfun/ggufit/main/ggufit.py
python3 ggufit.py ~/models/llama-3.1-8b-q4_k_m.gguf --gpu "RTX 4060"
```

```
Model: llama-3.1-8b-q4_k_m.gguf
  params: 8.00B   quant: Q4_K_M
  layers: 32   ctx: 4096
VRAM breakdown:
  weights:    4.19 GB
  KV cache:   0.50 GB  (0.12 GB / 1k tokens)
  logits:     1.96 GB
  overhead:   0.61 GB  (runtime estimate)
  total:      7.26 GB
GPU: NVIDIA GeForce RTX 4060
  total: 8.00 GB   usable: ~7.40 GB (after driver/OS reserve)
Verdict: TIGHT — only 0.14 GB headroom; one Chrome tab away from OOM
```

## The problem it solves

The usual workflow is backwards: download the model, watch the progress bar, let `failed to allocate 3412 MiB` do the math for you. The two numbers people forget are the **KV cache** (it grows with context length — a 32k context on an 8B model costs as much as the weights) and the fact that your 8 GB card does not have 8 GB free (driver, display server, OS take their cut first).

## Install

There is nothing to install. `ggufit.py` is one file, standard library only, Python 3.8+.

```sh
# option 1: just grab the file
curl -O https://raw.githubusercontent.com/techdevfun/ggufit/main/ggufit.py

# option 2: clone
git clone https://github.com/techdevfun/ggufit.git
```

## Usage

```sh
# a file you already have — reads only the header, instant even on 40 GB files
python3 ggufit.py model.gguf --gpu "RTX 4060"

# a file you DON'T have yet — the "should I even download this?" question
python3 ggufit.py --params 8B --quant q4_k_m --gpu "RTX 4060"

# longer context changes everything
python3 ggufit.py --params 8B --quant q4_k_m --ctx 32768 --gpu "RTX 4060"

# verdict against every card in the database
python3 ggufit.py --params 8B --quant q4_k_m --table

# your card isn't listed? give usable VRAM directly
python3 ggufit.py model.gguf --vram 10

# multiple cards at once
python3 ggufit.py model.gguf --gpu "4060" --gpu "M4"

# machine-readable, for scripts and CI
python3 ggufit.py model.gguf --gpu "RTX 4090" --json

# list the GPU database
python3 ggufit.py --list-gpus
```

When it won't fit, you get concrete next steps instead of a shrug:

```
Verdict: WON'T FIT — short by 3.36 GB
  try: shorten context to ~5160 tokens (7.39 GB total)
  try: offload ~80% of layers to CPU/RAM (keep ~6 of 32 on GPU)
```

Exit codes are script-friendly: `0` = fits (or an info command), `2` = won't fit, `1` = error. Chain it in front of your inference command so the doomed run never starts:

```sh
python3 ggufit.py model.gguf --gpu "RTX 4060" --json > /dev/null && llama-cli -m model.gguf
```

## How the math works

```
total = weights + KV cache + logits buffer + runtime overhead
weights   = sum over tensors of (params × bits-per-weight / 8), read from the file
KV cache  = 2 × layers × kv_heads × head_dim × context × 2 bytes (fp16)
logits    = vocab_size × hidden × 4 bytes
overhead  = ~0.4 GB CUDA/Metal context + ~5% compute buffers
usable    = card VRAM − reserve (driver/display/OS; larger haircut on unified memory)
```

Honest caveats, up front:

- **Weights are measured** (from the file's own tensor table). Everything else is a well-informed estimate — real usage shifts with the inference engine, flash attention, and batch size. Keep a safety margin.
- IQ/TQ quant types use average bits-per-weight; per-tensor imatrix files vary slightly. The report says so when it happens.
- The GPU database uses conservative *usable* VRAM, not the number on the box. If your setup differs, `--vram` overrides it.

## Why a CLI and not a website

Web calculators exist and they're fine. This one is for the terminal: no browser, no manual parameter entry, no uploading anything anywhere. It reads the file sitting on your disk and answers in milliseconds, which makes it usable *inside* scripts — as a pre-flight check before a download, a CI gate, or a guard in front of `llama-server`.

## Examples

`examples/` has a demo script and its captured output:

```sh
cd examples && sh run-demo.sh   # compare with expected-output.txt
```

`make_sample.py` writes a tiny synthetic GGUF so you can try file mode without downloading a real model.

## Roadmap

- [ ] Safetensors header support (same question, different format)
- [ ] `--kv-dtype` (quantized KV cache changes the math a lot)
- [ ] MoE-aware estimates (active vs. total params)
- [ ] Auto-detect the local GPU when possible (no dependency is the hard part)

## Contributing

Short version: open an issue before a big change, keep it dependency-free, add a test. Longer version in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
