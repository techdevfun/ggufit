#!/usr/bin/env python3
"""Write a tiny but structurally valid GGUF v3 file for demos and tests.

The model inside is a synthetic TOY (a few hundred thousand params) — it is
only here so you can try ggufit's file mode without downloading gigabytes.
The header layout (magic, KV metadata, tensor table) is real GGUF v3.
"""

import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _str(buf, s):
    b = s.encode("utf-8")
    return buf + struct.pack("<Q", len(b)) + b


def _u32(buf, v):
    return buf + struct.pack("<I", v)


def _u64(buf, v):
    return buf + struct.pack("<Q", v)


def _kv(buf, key, vtype, value):
    buf = _str(buf, key)
    buf = _u32(buf, vtype)
    if vtype == 4:  # uint32
        return buf + struct.pack("<I", value)
    if vtype == 8:  # string
        return _str(buf, value)
    if vtype == 9:  # array of strings
        buf = _u32(buf, 8)
        buf = _u64(buf, len(value))
        for s in value:
            buf = _str(buf, s)
        return buf
    raise AssertionError(vtype)


def main(out_path):
    arch = "llama"
    # A toy: 4 layers, hidden 256 — ~0.5M params. Honest numbers, tiny file.
    tensors = [
        ("tok_embeddings.weight", (1000, 256), 1),   # F16
        ("blk.0.attn_q.weight", (256, 256), 12),     # Q4_K
        ("blk.0.attn_k.weight", (128, 256), 12),     # Q4_K
        ("blk.0.ffn_up.weight", (512, 256), 2),      # Q4_0
        ("output.weight", (1000, 256), 1),           # F16
    ]
    buf = b"GGUF" + struct.pack("<I", 3)
    buf = _u64(buf, len(tensors))
    buf = _u64(buf, 7)
    buf = _kv(buf, "general.architecture", 8, arch)
    buf = _kv(buf, "general.name", 8, "ggufit-sample-toy")
    buf = _kv(buf, "%s.block_count" % arch, 4, 4)
    buf = _kv(buf, "%s.embedding_length" % arch, 4, 256)
    buf = _kv(buf, "%s.attention.head_count" % arch, 4, 4)
    buf = _kv(buf, "%s.attention.head_count_kv" % arch, 4, 2)
    buf = _kv(buf, "tokenizer.ggml.tokens", 9, ["<tok%d>" % i for i in range(1000)])
    offset = 0
    for name, dims, ttype in tensors:
        buf = _str(buf, name)
        buf = _u32(buf, len(dims))
        for d in dims:
            buf = _u64(buf, d)
        buf = _u32(buf, ttype)
        buf = _u64(buf, offset)
        offset += 4096  # dummy; ggufit never reads the weight body
    with open(out_path, "wb") as f:
        f.write(buf)
    print("wrote %s (%d bytes)" % (out_path, len(buf)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sample.gguf"))
