#!/bin/bash
cd /workspace/codi-circuit
source /workspace/codi-circuit/venv/bin/activate
python visualize_llama_circuits.py \
  --model /workspace/codi-circuit/my_models/merged_bcywinski_codi_llama1b-answer_only \
  --prompt "A team starts with 3 members. 2 members leave the team. Then each remaining member recruits 4 additional people. How many people are there now on the team? Give the answer only and nothing else." \
  --max-new-tokens 256 \
  --overwrite \
  --offload disk \
  --batch-size 64 \
  --backend transformerlens \
  --max-n-logits 2 \
  --verbose \
  --max-feature-nodes 500
