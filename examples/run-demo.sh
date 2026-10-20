#!/bin/sh
# Demo for ggufit. Everything runs offline; nothing is downloaded.
# Scenario 1 uses a synthetic toy GGUF (see make_sample.py) so the repo
# stays small. Scenarios 2-4 use --params mode: the "I haven't downloaded
# it yet, will it fit?" question.
set -e
cd "$(dirname "$0")"

echo "=== 1. file mode: read a real GGUF header (synthetic toy file) ==="
python3 make_sample.py sample.gguf
python3 ../ggufit.py sample.gguf --gpu "RTX 4060"

echo ""
echo "=== 2. manual mode: 8B Q4_K_M on an RTX 4060 (the classic question) ==="
python3 ../ggufit.py --params 8B --quant q4_k_m --gpu "RTX 4060" || true

echo ""
echo "=== 3. manual mode: 70B Q4_K_M on an RTX 4090 (with suggestions) ==="
python3 ../ggufit.py --params 70B --quant q4_k_m --gpu "RTX 4090" || true

echo ""
echo "=== 4. verdict table: 8B Q4_K_M against every card in the DB ==="
python3 ../ggufit.py --params 8B --quant q4_k_m --table | head -20
