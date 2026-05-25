#!/bin/bash

PRED_PATH="Output/1_edited_hybrid/all_results.json"

EVAL_SCRIPT="./evaluation.py"

python3 "$EVAL_SCRIPT" \
  --input "$PRED_PATH" \


