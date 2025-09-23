## ProtoQA decoder benchmark

```bash
# create env and install deps
python -m venv conceptEnv
source conceptEnv/bin/activate
pip install -r requirements.txt

# run (from repo root)
python concept-decoder_protoqa/protoqa_decoder_benchmark.py \
  --model "Qwen/Qwen3-0.6B" \
  --protoqa_dir "./protoqa-data/data" \
  --split dev \
  --top_k_tokens 100 \
  --out "./results/protoqa_decoder_dev.csv"