#!/bin/bash
PYTHON_EXEC="python"

DATASET_NAME="MQuAKE-T" # "MQuAKE-T" or "MQuAKE-CF-3k-v2"

# Input files
KNOWLEDGE_FILE="datasets/knowledge/${DATASET_NAME}_knowledge.json"
QUESTION_FILE="datasets/questions/${DATASET_NAME}_questions.json"


OUTPUT_BASE_DIR="Output"
BATCH_SIZE="1" # 1, 100, "All"
START_INDEX=1 # from 1
QUERY_MODE="hybrid" # "lm", "gnn" or "hybrid"
PRETRAIN="--pretrain" 
DEVICE=1 # GPU device ID
GNN_DIMENSION=256 # GNN embedding dimension, e.g., 128, 256, 512.


$PYTHON_EXEC full_pipe.py \
    --knowledge_file "datasets/knowledge/${DATASET_NAME}_knowledge.json" \
    --question_file "datasets/questions/${DATASET_NAME}_questions.json" \
    --output_base_dir $OUTPUT_BASE_DIR \
    --batch_size $BATCH_SIZE \
    --start_index $START_INDEX \
    --query_mode $QUERY_MODE \
    $PRETRAIN \
    --device $DEVICE \
    --GNN_dimension $GNN_DIMENSION