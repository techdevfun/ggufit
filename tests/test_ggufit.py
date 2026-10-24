"""Tests for ggufit. Builds tiny synthetic GGUF files in a temp dir —
no network, no real model needed."""

import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ggufit

BIN = [os.path.join(os.path.dirname(__file__), "..", "ggufit.py")]


def w_str(buf, s):
    b = s.encode("utf-8")
    return buf + struct.pack("<Q", len(b)) + b


def w_u32(buf, v):
    return buf + struct.pack("<I", v)


def w_u64(buf, v):
    return buf + struct.pack("<Q", v)


def w_val(buf, vtype, v):
    buf = w_u32(buf, vtype)
    if vtype == 4:  # uint32
        return buf + struct.pack("<I", v)
    if vtype == 8:  # string
        return w_str(buf, v)
    if vtype == 9:  # array of strings
        buf = w_u32(buf, 8)
        buf = w_u64(buf, len(v))
        for s in v:
            buf = w_str(buf, s)
        return buf
    raise AssertionError("test writer: unsupported type %d" % vtype)


def make_gguf(path, arch="llama", tensors=((12, (256,)), (1, (64,))),
              block_count=2, hidden=64, heads=4, kv_heads=2, vocab=100):
    """Write a minimal valid GGUF v3 file. Default: one Q4_K (256-elem) and
    one F16 (64-elem) tensor."""
    buf = b"GGUF" + struct.pack("<I", 3)
    kv = [
        ("general.architecture", 8, arch),
        ("%s.block_count" % arch, 4, block_count),
        ("%s.embedding_length" % arch, 4, hidden),
        ("%s.attention.head_count" % arch, 4, heads),
        ("%s.attention.head_count_kv" % arch, 4, kv_heads),
        ("tokenizer.ggml.tokens", 9, ["tok%d" % i for i in range(vocab)]),
    ]
    buf = w_u64(buf, len(tensors))
    buf = w_u64(buf, len(kv))
    for k, t, v in kv:
        buf = w_str(buf, k)
        buf = w_val(buf, t, v)
    for i, (ttype, dims) in enumerate(tensors):
        buf = w_str(buf, "tensor.%d" % i)
        buf = w_u32(buf, len(dims))
        for d in dims:
            buf = w_u64(buf, d)
        buf = w_u32(buf, ttype)
        buf = w_u64(buf, 0)  # offset unused by parser
    with open(path, "wb") as f:
        f.write(buf)


class GgufParseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gguf = os.path.join(self.tmp, "tiny.gguf")
        make_gguf(self.gguf)

    def test_params_and_weights_from_header(self):
        meta, tensors = ggufit.GgufReader(self.gguf).parse()
        self.assertEqual(meta["general.architecture"], "llama")
        self.assertEqual(len(tensors), 2)

    def test_model_math(self):
        m = ggufit.model_from_gguf(self.gguf)
        # 256 + 64 params; Q4_K = 4.5bpw -> 144 B, F16 = 128 B
        self.assertEqual(m["n_params"], 320)
        self.assertAlmostEqual(m["weight_bytes"], 144 + 128)
        self.assertEqual(m["layers"], 2)
        self.assertEqual(m["kv_heads"], 2)

    def test_kv_cache_math(self):
        m = ggufit.model_from_gguf(self.gguf)
        est = ggufit.estimate(m, ctx=1024)
        # 2 * layers(2) * kv_heads(2) * head_dim(16) * 1024 * 2 bytes
        self.assertEqual(est["kv_cache"], 2 * 2 * 2 * 16 * 1024 * 2)
        # logits: vocab(100) * hidden(64) * 4
        self.assertEqual(est["logits"], 100 * 64 * 4)

    def test_bad_magic(self):
        bad = os.path.join(self.tmp, "bad.gguf")
        with open(bad, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 32)
        with self.assertRaises(ggufit.GgufError):
            ggufit.model_from_gguf(bad)

    def test_missing_arch_metadata(self):
        p = os.path.join(self.tmp, "noarch.gguf")
        buf = b"GGUF" + struct.pack("<I", 3) + w_u64(b"", 0) + w_u64(b"", 0)
        with open(p, "wb") as f:
            f.write(buf)
        with self.assertRaises(ggufit.GgufError):
            ggufit.model_from_gguf(p)

    def test_unknown_quant_type_falls_back(self):
        p = os.path.join(self.tmp, "weird.gguf")
        make_gguf(p, tensors=((99, (256,)),))
        m = ggufit.model_from_gguf(p)
        self.assertTrue(m["quant_approx"])
        self.assertAlmostEqual(m["weight_bytes"], 256 * 4.5 / 8)


class VerdictTest(unittest.TestCase):
    def test_fits_tight_wontfit(self):
        self.assertEqual(ggufit.verdict_for(5 * ggufit.GB, 8 * ggufit.GB)[0], "FITS")
        self.assertEqual(ggufit.verdict_for(7.5 * ggufit.GB, 8 * ggufit.GB)[0], "TIGHT")
        self.assertEqual(ggufit.verdict_for(9 * ggufit.GB, 8 * ggufit.GB)[0], "WON'T FIT")

    def test_suggest_smaller_quant(self):
        m = ggufit.model_from_manual("8B", "q8_0")
        s = ggufit.suggest(m, 4096, 8 * ggufit.GB)
        self.assertTrue(any("q4" in x or "q3" in x or "q2" in x for x in s),
                        "expected a smaller-quant suggestion, got %r" % (s,))

    def test_suggest_offload(self):
        m = ggufit.model_from_manual("70B", "q4_k_m")
        s = ggufit.suggest(m, 4096, 24 * ggufit.GB)
        self.assertTrue(any("offload" in x for x in s), "got %r" % (s,))


class ManualModeTest(unittest.TestCase):
    def test_params_parsing(self):
        self.assertEqual(ggufit.parse_params("8B"), 8_000_000_000)
        self.assertEqual(ggufit.parse_params("1.5B"), 1_500_000_000)
        self.assertEqual(ggufit.parse_params("700M"), 700_000_000)

    def test_quant_aliases(self):
        self.assertEqual(ggufit.norm_quant("Q4_K_M"), "q4_k")
        self.assertEqual(ggufit.norm_quant("q80"), "q8_0")
        with self.assertRaises(ggufit.GgufError):
            ggufit.norm_quant("q99_x")

    def test_preset_used(self):
        m = ggufit.model_from_manual("8B", "q4_k_m")
        self.assertTrue(m["arch_known"])
        self.assertEqual(m["layers"], 32)
        self.assertAlmostEqual(m["weight_bytes"], 8e9 * 4.5 / 8, delta=1e6)

    def test_no_preset_estimated(self):
        m = ggufit.model_from_manual("3B", "q4_k_m")
        self.assertFalse(m["arch_known"])


class GpuDbTest(unittest.TestCase):
    def test_exact_and_substring(self):
        self.assertEqual(ggufit.find_gpu("RTX 4060")[0], "NVIDIA GeForce RTX 4060")
        self.assertEqual(ggufit.find_gpu("m4")[0], "Apple M4 (unified)")

    def test_ambiguous(self):
        with self.assertRaises(ggufit.GgufError):
            ggufit.find_gpu("RTX")

    def test_unknown(self):
        self.assertIsNone(ggufit.find_gpu("RTX 9999"))

    def test_usable_less_than_total(self):
        for name, total, reserve in ggufit.GPUS:
            self.assertLess(total - reserve, total, name)
            self.assertGreater(total - reserve, 0, name)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, BIN[0], *args],
                              capture_output=True, text=True, timeout=60)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gguf = os.path.join(self.tmp, "tiny.gguf")
        make_gguf(self.gguf)

    def test_file_mode_fits(self):
        r = self.run_cli(self.gguf, "--gpu", "RTX 4090")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("FITS", r.stdout)

    def test_manual_mode_wont_fit_exit_2(self):
        r = self.run_cli("--params", "70B", "--quant", "q4_k_m",
                         "--gpu", "RTX 4090")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("WON'T FIT", r.stdout)
        self.assertIn("try:", r.stdout)
        self.assertIn("offload", r.stdout)

    def test_json_valid(self):
        r = self.run_cli("--params", "8B", "--quant", "q4_k_m",
                         "--gpu", "RTX 4060", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertIn("total_gb", data)
        self.assertEqual(data["targets"][0]["gpu"], "NVIDIA GeForce RTX 4060")

    def test_json_exit_2_when_any_target_wont_fit(self):
        r = self.run_cli("--params", "70B", "--quant", "q4_k_m",
                         "--gpu", "RTX 4090", "--gpu", "A100 80GB", "--json")
        self.assertEqual(r.returncode, 2, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual([t["verdict"] for t in data["targets"]],
                         ["WON'T FIT", "FITS"])

    def test_json_exit_0_when_all_targets_fit(self):
        r = self.run_cli("--params", "8B", "--quant", "q4_k_m",
                         "--gpu", "RTX 4060", "--gpu", "RTX 4090", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertNotIn("WON'T FIT", [t["verdict"] for t in data["targets"]])

    def test_table(self):
        r = self.run_cli("--params", "8B", "--quant", "q4_k_m", "--table")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RTX 4090", r.stdout)

    def test_list_gpus(self):
        r = self.run_cli("--list-gpus")
        self.assertEqual(r.returncode, 0)
        self.assertIn("usable", r.stdout)

    def test_no_gpu_errors(self):
        r = self.run_cli("--params", "8B", "--quant", "q4_k_m")
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
