# Contributing to ggufit

## Ground rules

1. **Zero dependencies, forever.** Standard library only. If a feature needs a
   package, it doesn't ship.
2. **One file.** `ggufit.py` stays a single file you can `curl`. Keep it
   readable: small functions, no cleverness.
3. **Honest numbers.** If a value is estimated rather than measured, the output
   must say so. Never present a guess as a fact.

## Workflow

- Open an issue first for anything bigger than a typo — describe the problem,
  not just the patch.
- Add a test in `tests/` for behavior changes. Run `python3 -m unittest discover -s tests`.
- New GPU in the database? Total VRAM from the vendor spec sheet; keep the
  reserve conservative and consistent with similar cards.
- New quant type? Bits-per-weight from ggml block sizes (exact) or published
  averages (mark approximate in `BPW`).

## Style

- Python 3.8-compatible syntax (no `match`, no `X | Y` annotations).
- CLI output stays plain text — it must survive being piped, logged, and
  read over SSH.
