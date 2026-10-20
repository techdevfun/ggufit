#!/usr/bin/env python3
"""ggufit — will this GGUF model actually fit on your GPU?

Reads a local .gguf file's real header (no upload, no network, no
dependencies) and answers the question everyone asks *after* downloading
4 GB: do the weights + KV cache + runtime overhead fit in my card's VRAM?

Quick start:
    python3 ggufit.py model.gguf --gpu "RTX 4060"
    python3 ggufit.py --params 8B --quant q4_k_m --gpu "RTX 4060"   # not downloaded yet?
    python3 ggufit.py model.gguf --table                            # every card in the DB

Exit codes: 0 = fits (or an info command), 2 = won't fit, 1 = error.
"""

import argparse
import json
import struct
import sys

VERSION = "1.0.0"
GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Bits per weight for each ggml quant type.
# Exact for the mainstream types (from ggml block sizes); approximate averages
# for the IQ/TQ/MXFP4 family (per-tensor imatrix makes these vary slightly).
# Values nobody should bet their life on are flagged "approximate" in output.
# ---------------------------------------------------------------------------
BPW = {
    # name: (bits_per_weight, approximate?)
    "f32": (32.0, False),
    "f16": (16.0, False),
    "bf16": (16.0, False),
    "q8_0": (8.5, False),
    "q6_k": (6.5625, False),
    "q5_k": (5.5, False),
    "q5_0": (5.5, False),
    "q5_1": (6.0, False),
    "q4_k": (4.5, False),
    "q4_0": (4.5, False),
    "q4_1": (5.0, False),
    "q3_k": (3.4375, False),
    "q2_k": (2.625, False),
    "i8": (8.0, False),
    "i16": (16.0, False),
    "i32": (32.0, False),
    "iq4_nl": (4.5, True),
    "iq4_xs": (4.25, True),
    "iq3_xxs": (3.06, True),
    "iq3_s": (3.44, True),
    "iq2_xxs": (2.06, True),
    "iq2_xs": (2.31, True),
    "iq2_s": (2.5, True),
    "iq1_s": (1.56, True),
    "iq1_m": (1.75, True),
    "tq1_0": (2.06, True),
    "tq2_0": (2.31, True),
    "mxfp4": (4.5, True),
}

# ggml type id -> canonical name (only the ones files actually contain)
GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k",
    14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
    19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4",
}
BPW["q8_k"] = (9.0625, False)
BPW["i64"] = (64.0, False)
BPW["f64"] = (64.0, False)

QUANT_ALIASES = {
    "q4_k_m": "q4_k", "q4_k_s": "q4_k", "q4k": "q4_k",
    "q5_k_m": "q5_k", "q5_k_s": "q5_k", "q5k": "q5_k",
    "q6_k_m": "q6_k", "q3_k_m": "q3_k", "q3_k_s": "q3_k", "q3_k_l": "q3_k",
    "q2_k_m": "q2_k", "q2_k_s": "q2_k",
    "q8": "q8_0", "q80": "q8_0",
    "fp32": "f32", "fp16": "f16",
    "iq4_nl": "iq4_nl", "iq4nl": "iq4_nl",
}

# Quant ladder, best quality first — used for "what smaller quant would fit?"
QUANT_LADDER = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q4_0",
                "q3_k", "iq4_xs", "q2_k", "iq2_xs"]

# ---------------------------------------------------------------------------
# GPU database: usable VRAM, not marketing VRAM.
# reserve_gb is subtracted for driver / display / OS. Unified-memory machines
# (Apple) share RAM with the OS, so they get a larger haircut.
# Sources: vendor spec sheets for total VRAM; reserves are conservative rules
# of thumb, documented in README. If a card is missing, --vram N works too.
# ---------------------------------------------------------------------------
GPUS = [
    # (display name, total GB, reserve GB)
    ("NVIDIA GeForce RTX 5090", 32, 0.6),
    ("NVIDIA GeForce RTX 5080", 16, 0.6),
    ("NVIDIA GeForce RTX 5070 Ti", 16, 0.6),
    ("NVIDIA GeForce RTX 5070", 12, 0.6),
    ("NVIDIA GeForce RTX 5060 Ti 16GB", 16, 0.6),
    ("NVIDIA GeForce RTX 5060", 8, 0.6),
    ("NVIDIA GeForce RTX 4090", 24, 0.6),
    ("NVIDIA GeForce RTX 4080 Super", 16, 0.6),
    ("NVIDIA GeForce RTX 4080", 16, 0.6),
    ("NVIDIA GeForce RTX 4070 Ti Super", 16, 0.6),
    ("NVIDIA GeForce RTX 4070 Super", 12, 0.6),
    ("NVIDIA GeForce RTX 4070", 12, 0.6),
    ("NVIDIA GeForce RTX 4060 Ti 16GB", 16, 0.6),
    ("NVIDIA GeForce RTX 4060 Ti 8GB", 8, 0.6),
    ("NVIDIA GeForce RTX 4060", 8, 0.6),
    ("NVIDIA GeForce RTX 3090", 24, 0.6),
    ("NVIDIA GeForce RTX 3080 12GB", 12, 0.6),
    ("NVIDIA GeForce RTX 3080 10GB", 10, 0.6),
    ("NVIDIA GeForce RTX 3070", 8, 0.6),
    ("NVIDIA GeForce RTX 3060 Ti", 8, 0.6),
    ("NVIDIA GeForce RTX 3060 12GB", 12, 0.6),
    ("NVIDIA GeForce RTX 3050 8GB", 8, 0.6),
    ("NVIDIA GeForce RTX 3050 6GB", 6, 0.6),
    ("NVIDIA GeForce RTX 4060 Laptop", 8, 0.8),
    ("NVIDIA GeForce RTX 4070 Laptop", 8, 0.8),
    ("NVIDIA GeForce GTX 1660 Super", 6, 0.6),
    ("NVIDIA Tesla T4", 16, 0.3),
    ("NVIDIA L40S", 48, 0.3),
    ("NVIDIA A100 80GB", 80, 0.5),
    ("NVIDIA A100 40GB", 40, 0.5),
    ("NVIDIA H100 80GB", 80, 0.5),
    ("AMD Radeon RX 7900 XTX", 24, 0.6),
    ("AMD Radeon RX 7900 XT", 20, 0.6),
    ("AMD Radeon RX 7800 XT", 16, 0.6),
    ("AMD Radeon RX 7700 XT", 12, 0.6),
    ("AMD Radeon RX 7600", 8, 0.6),
    ("AMD Radeon RX 6600", 8, 0.6),
    ("AMD Radeon RX 9070 XT", 16, 0.6),
    ("AMD Radeon RX 9070", 16, 0.6),
    ("Intel Arc A770", 16, 0.6),
    ("Intel Arc B580", 12, 0.6),
    ("Apple M4 (unified)", 16, 4.8),
    ("Apple M3 Pro (unified)", 18, 5.4),
    ("Apple M2 (unified)", 16, 4.8),
    ("Apple M1 (unified)", 16, 4.8),
]

# Common dense architectures for --params mode (layers, hidden, heads, kv_heads, vocab)
ARCH_PRESETS = {
    7: (32, 4096, 32, 32, 32000),
    8: (32, 4096, 32, 8, 128256),
    9: (42, 3584, 16, 8, 256000),
    13: (40, 5120, 40, 40, 32000),
    14: (48, 5120, 40, 8, 152064),
    27: (46, 4608, 32, 16, 256000),
    32: (64, 5120, 40, 8, 152064),
    70: (80, 8192, 64, 8, 128256),
}


class GgufError(Exception):
    pass


# ---------------------------------------------------------------------------
# Minimal GGUF v3 header reader. Only the header is ever read — the weight
# body is never touched, so this is instant even on 40 GB files.
# ---------------------------------------------------------------------------
class GgufReader:
    def __init__(self, path, header_limit=64 * 1024 * 1024):
        self.f = open(path, "rb")
        self.limit = header_limit

    def _read(self, n):
        data = self.f.read(n)
        if len(data) != n:
            raise GgufError("unexpected end of file while reading GGUF header")
        return data

    def _u8(self):
        return struct.unpack("<B", self._read(1))[0]

    def _u32(self):
        return struct.unpack("<I", self._read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._read(8))[0]

    def _str(self):
        n = self._u64()
        if n > self.limit:
            raise GgufError("suspicious string length %d in GGUF header" % n)
        return self._read(n).decode("utf-8", errors="replace")

    def _value(self, vtype):
        if vtype == 0:
            return self._u8()
        if vtype == 1:
            return struct.unpack("<b", self._read(1))[0]
        if vtype == 2:
            return struct.unpack("<H", self._read(2))[0]
        if vtype == 3:
            return struct.unpack("<h", self._read(2))[0]
        if vtype == 4:
            return self._u32()
        if vtype == 5:
            return struct.unpack("<i", self._read(4))[0]
        if vtype == 6:
            return struct.unpack("<f", self._read(4))[0]
        if vtype == 7:
            return bool(self._u8())
        if vtype == 8:
            return self._str()
        if vtype == 9:
            elem_type = self._u32()
            n = self._u64()
            if n > 10_000_000:
                raise GgufError("suspicious array length %d in GGUF header" % n)
            return [self._value(elem_type) for _ in range(n)]
        if vtype == 10:
            return self._u64()
        if vtype == 11:
            return struct.unpack("<q", self._read(8))[0]
        if vtype == 12:
            return struct.unpack("<d", self._read(8))[0]
        raise GgufError("unknown GGUF metadata value type %d" % vtype)

    def parse(self):
        if self._read(4) != b"GGUF":
            raise GgufError("not a GGUF file (bad magic)")
        version = self._u32()
        if version not in (2, 3):
            raise GgufError("unsupported GGUF version %d (expected 2 or 3)" % version)
        n_tensors = self._u64()
        n_kv = self._u64()
        if n_tensors > 10_000_000 or n_kv > 10_000_000:
            raise GgufError("suspicious GGUF header counts")
        meta = {}
        for _ in range(n_kv):
            key = self._str()
            vtype = self._u32()
            meta[key] = self._value(vtype)
        tensors = []
        for _ in range(n_tensors):
            name = self._str()
            n_dims = self._u32()
            if n_dims > 8:
                raise GgufError("suspicious tensor rank %d" % n_dims)
            dims = tuple(self._u64() for _ in range(n_dims))
            ttype = self._u32()
            offset = self._u64()
            tensors.append((name, dims, ttype, offset))
        self.f.close()
        return meta, tensors


def norm_quant(name):
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = QUANT_ALIASES.get(key, key)
    if key not in BPW:
        raise GgufError("unknown quantization '%s'" % name)
    return key


def parse_params(text):
    t = text.strip().lower().replace(" ", "")
    mult = 1
    if t.endswith("b"):
        mult, t = 1_000_000_000, t[:-1]
    elif t.endswith("m"):
        mult, t = 1_000_000, t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        raise GgufError("can't parse parameter count '%s' (try 8B, 1.5B, 700M)" % text)


def find_gpu(query):
    q = query.strip().lower().replace("-", " ").replace("_", " ")
    hits = [g for g in GPUS if q in g[0].lower().replace("-", " ")]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # Prefer an exact-ish hit (query covers the whole name minus vendor words)
    exact = [g for g in hits if g[0].lower().replace("nvidia geforce ", "")
             .replace("amd radeon ", "").replace("intel arc ", "") == q
             or g[0].lower() == q]
    if len(exact) == 1:
        return exact[0]
    raise GgufError(
        "ambiguous GPU '%s' — matches %d cards: %s"
        % (query, len(hits), ", ".join(g[0] for g in hits[:8]))
    )


def model_from_gguf(path):
    """Build a model spec from a real GGUF file header."""
    meta, tensors = GgufReader(path).parse()
    arch = meta.get("general.architecture")
    n_params = 0
    weight_bytes = 0.0
    quant_names = {}
    approx = False
    for _name, dims, ttype, _off in tensors:
        n_elem = 1
        for d in dims:
            n_elem *= d
        qname = GGML_TYPE_NAMES.get(ttype)
        if qname is None:
            # Unknown type: assume ~4.5 bpw and say so loudly.
            bpw, is_approx = 4.5, True
        else:
            bpw, is_approx = BPW[qname]
        approx = approx or is_approx
        n_params += n_elem
        weight_bytes += n_elem * bpw / 8.0
        qkey = qname or ("type_%d" % ttype)
        quant_names[qkey] = quant_names.get(qkey, 0) + n_elem
    if n_params == 0:
        raise GgufError("GGUF file contains no tensors")

    if arch:
        layers = meta.get("%s.block_count" % arch)
        hidden = meta.get("%s.embedding_length" % arch)
        heads = meta.get("%s.attention.head_count" % arch)
        kv_heads = meta.get("%s.attention.head_count_kv" % arch)
        if layers is None or hidden is None:
            raise GgufError(
                "GGUF header is missing architecture metadata "
                "(%s.block_count / embedding_length); is the file complete?" % arch)
    else:
        layers = hidden = heads = kv_heads = None

    tokens = meta.get("tokenizer.ggml.tokens")
    vocab = len(tokens) if isinstance(tokens, list) else None
    quant_label = meta.get("general.type") or _dominant_quant(quant_names)
    return {
        "name": path.split("/")[-1],
        "n_params": n_params,
        "weight_bytes": weight_bytes,
        "quant_label": str(quant_label),
        "quant_approx": approx,
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "vocab": vocab,
        "arch_known": arch is not None and layers is not None,
    }


def _dominant_quant(quant_names):
    if not quant_names:
        return "?"
    return max(quant_names, key=quant_names.get)


def model_from_manual(params_text, quant_text, layers=None, hidden=None,
                      heads=None, kv_heads=None, vocab=None):
    n_params = parse_params(params_text)
    quant = norm_quant(quant_text)
    bpw, approx = BPW[quant]
    arch_known = True
    if layers is None or hidden is None:
        preset = ARCH_PRESETS.get(round(n_params / 1e9))
        if preset:
            layers, hidden, heads, kv_heads, vocab = preset
        else:
            # Generic dense-model guess. Labeled estimated everywhere it shows.
            layers, hidden = 32, 4096
            heads, kv_heads = 32, 8
            vocab = 128256
            arch_known = False
    kv_heads = kv_heads or (heads // 4 if heads else 8)
    return {
        "name": "%s %s (manual)" % (params_text, quant_text),
        "n_params": n_params,
        "weight_bytes": n_params * bpw / 8.0,
        "quant_label": quant,
        "quant_approx": approx,
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "vocab": vocab,
        "arch_known": arch_known,
    }


def estimate(model, ctx, kv_dtype_bytes=2):
    """VRAM breakdown in bytes. Honest about what is measured vs estimated."""
    weights = model["weight_bytes"]
    layers = model["layers"] or 32
    hidden = model["hidden"] or 4096
    heads = model["heads"] or 32
    kv_heads = model["kv_heads"] or max(1, heads // 4)
    head_dim = hidden // heads if heads else 128
    kv_cache = 2 * layers * kv_heads * head_dim * ctx * kv_dtype_bytes
    vocab = model["vocab"]
    logits = (vocab * hidden * 4) if vocab else 0.0
    # CUDA/Metal context + compute buffers: a rule of thumb, not a measurement.
    overhead = 0.4 * GB + 0.05 * weights
    total = weights + kv_cache + logits + overhead
    return {
        "weights": weights,
        "kv_cache": kv_cache,
        "logits": logits,
        "overhead": overhead,
        "total": total,
        "ctx": ctx,
        "kv_per_1k": 2 * layers * kv_heads * head_dim * 1024 * kv_dtype_bytes,
    }


def verdict_for(total, usable):
    headroom = usable - total
    if headroom >= 0.10 * usable:
        return "FITS", headroom
    if headroom >= 0:
        return "TIGHT", headroom
    return "WON'T FIT", headroom


def fmt_gb(b):
    return "%.2f GB" % (b / GB)


def suggest(model, ctx, usable):
    """What to change when it doesn't fit. All suggestions are estimates."""
    out = []
    base = estimate(model, ctx)
    # 1. Best smaller-or-equal quant that fits (walk down from current).
    cur = model["quant_label"]
    ladder = QUANT_LADDER
    start = ladder.index(cur) if cur in ladder else len(ladder) - 1
    for q in ladder[start + 1:]:
        m2 = dict(model)
        m2["weight_bytes"] = model["n_params"] * BPW[q][0] / 8.0
        m2["quant_label"] = q
        if estimate(m2, ctx)["total"] <= usable:
            out.append("drop to %s (%s total)" % (q, fmt_gb(estimate(m2, ctx)["total"])))
            break
    # 2. Largest context that fits at the current quant.
    lo, hi, best = 256, ctx, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if estimate(model, mid)["total"] <= usable:
            best, lo = mid, mid + 256
        else:
            hi = mid - 256
    if 0 < best < ctx:
        out.append("shorten context to ~%d tokens (%s total)"
                   % (best, fmt_gb(estimate(model, best)["total"])))
    # 3. Partial GPU offload: keep this share of weights on the GPU.
    rest = base["total"] - base["weights"]
    if base["weights"] > 0 and rest < usable:
        keep = (usable - rest) / base["weights"]
        if 0 < keep < 1:
            layers = model["layers"] or 32
            out.append("offload ~%d%% of layers to CPU/RAM (keep ~%d of %d on GPU)"
                       % (round(100 * (1 - keep)), int(layers * keep), layers))
    return out


def report_text(model, est, gpu_name, total_gb, usable_gb):
    verdict, headroom = verdict_for(est["total"], usable_gb * GB)
    lines = []
    lines.append("Model: %s" % model["name"])
    lines.append("  params: %.2fB   quant: %s%s" % (
        model["n_params"] / 1e9, model["quant_label"],
        "  (weights estimated)" if model["quant_approx"] else ""))
    arch_note = "" if model["arch_known"] else "  (architecture estimated)"
    lines.append("  layers: %s   ctx: %d%s" % (model["layers"], est["ctx"], arch_note))
    lines.append("VRAM breakdown:")
    lines.append("  weights:    %s" % fmt_gb(est["weights"]))
    lines.append("  KV cache:   %s  (%s / 1k tokens)" % (fmt_gb(est["kv_cache"]), fmt_gb(est["kv_per_1k"])))
    if est["logits"]:
        lines.append("  logits:     %s" % fmt_gb(est["logits"]))
    lines.append("  overhead:   %s  (runtime estimate)" % fmt_gb(est["overhead"]))
    lines.append("  total:      %s" % fmt_gb(est["total"]))
    lines.append("GPU: %s" % gpu_name)
    lines.append("  total: %s   usable: ~%s (after driver/OS reserve)"
                 % (fmt_gb(total_gb * GB), fmt_gb(usable_gb * GB)))
    if verdict == "FITS":
        lines.append("Verdict: FITS — %s headroom" % fmt_gb(headroom))
    elif verdict == "TIGHT":
        lines.append("Verdict: TIGHT — only %s headroom; one Chrome tab away from OOM"
                     % fmt_gb(headroom))
    else:
        lines.append("Verdict: WON'T FIT — short by %s" % fmt_gb(-headroom))
        for s in suggest(model, est["ctx"], usable_gb * GB):
            lines.append("  try: " + s)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ggufit",
        description="Will this GGUF model fit on your GPU? Offline, zero-dependency.")
    ap.add_argument("file", nargs="?", help="local .gguf file to inspect")
    ap.add_argument("--gpu", action="append", default=[],
                    help="GPU to check against (repeatable; substring match)")
    ap.add_argument("--vram", type=float, default=None,
                    help="usable VRAM in GB (when your card isn't in the DB)")
    ap.add_argument("--ctx", type=int, default=4096, help="context length (default 4096)")
    ap.add_argument("--params", help="manual mode: parameter count, e.g. 8B")
    ap.add_argument("--quant", help="manual mode: quantization, e.g. q4_k_m")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--kv-heads", type=int, default=None)
    ap.add_argument("--list-gpus", action="store_true", help="list the GPU database")
    ap.add_argument("--table", action="store_true",
                    help="verdict against every GPU in the database")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)

    if args.version:
        print("ggufit " + VERSION)
        return 0
    if args.list_gpus:
        for name, total, reserve in sorted(GPUS, key=lambda g: g[1]):
            print("%-32s %5.1f GB total  ~%4.1f GB usable"
                  % (name, total, total - reserve))
        return 0

    try:
        if args.file:
            model = model_from_gguf(args.file)
        elif args.params and args.quant:
            model = model_from_manual(args.params, args.quant, args.layers,
                                      args.hidden, None, args.kv_heads)
        else:
            ap.error("give me a .gguf file, or --params + --quant")
    except GgufError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1

    est = estimate(model, args.ctx)

    targets = []
    if args.vram:
        targets.append(("custom (%.1f GB usable)" % args.vram, args.vram, args.vram))
    for q in args.gpu:
        hit = find_gpu(q)
        if hit is None:
            print("error: no GPU matches '%s' (try --list-gpus)" % q, file=sys.stderr)
            return 1
        name, total, reserve = hit
        targets.append((name, total, total - reserve))
    if args.table:
        targets = [(n, t, t - r) for n, t, r in sorted(GPUS, key=lambda g: g[1])]
    if not targets:
        print("error: no GPU given — use --gpu, --vram, or --table", file=sys.stderr)
        return 1

    if args.json:
        rows = []
        for name, total, usable in targets:
            v, headroom = verdict_for(est["total"], usable * GB)
            rows.append({
                "gpu": name, "vram_total_gb": total, "vram_usable_gb": round(usable, 2),
                "verdict": v, "headroom_gb": round(headroom / GB, 2),
            })
        print(json.dumps({
            "model": model["name"], "params_b": round(model["n_params"] / 1e9, 2),
            "quant": model["quant_label"], "ctx": est["ctx"],
            "weights_gb": round(est["weights"] / GB, 2),
            "kv_cache_gb": round(est["kv_cache"] / GB, 2),
            "overhead_gb": round(est["overhead"] / GB, 2),
            "total_gb": round(est["total"] / GB, 2),
            "targets": rows,
        }, indent=2))
        worst = min(r["verdict"] for r in rows)
        return 2 if worst == "WON'T FIT" else 0

    if args.table:
        print("%-32s %8s %8s  %s" % ("GPU", "usable", "total", "verdict"))
        for name, total, usable in targets:
            v, _ = verdict_for(est["total"], usable * GB)
            print("%-32s %7.1fG %7.1fG  %s" % (name, usable, total, v))
        return 0

    rc = 0
    for i, (name, total, usable) in enumerate(targets):
        if i:
            print()
        print(report_text(model, est, name, total, usable))
        v, _ = verdict_for(est["total"], usable * GB)
        if v == "WON'T FIT":
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
