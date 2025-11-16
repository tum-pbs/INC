#!/bin/bash
# chmod +x train_BFS.sh

DEFAULT_GPU=1
MODE="train"
METHOD="INC"
MODEL_TYPE="SmallCNN"
SUBSTEPS=1
MODEL_ID="" 

# --- Script Logic ---

# Function to display how to use the script
show_usage() {
    echo "Usage: $0 [gpu_id]"
    echo ""
    echo "  [gpu_id] (optional): The ID of the GPU to use. Defaults to $DEFAULT_GPU."
    echo ""
    echo "All other settings are hardcoded within the script:"
    echo "  Mode:         $MODE"
    echo "  Method:       $METHOD"
    echo "  Model type:   $MODEL_TYPE"
    echo "  Substeps:     $SUBSTEPS"
    [ -n "$MODEL_ID" ] && echo "  Model ID:     $MODEL_ID"
    exit 1
}

# --- Main argument parsing ---
# The script now only accepts one optional argument for the GPU ID.
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
fi

# Set the GPU ID from the first argument, or use the default value.
GPU_ID=${1:-$DEFAULT_GPU}

# Prepare output directory
today=$(date +%Y-%m-%d)
mkdir -p "./out/$today"

# Generate filename based on the hardcoded settings
model_id_part=""
[ -n "$MODEL_ID" ] && model_id_part="_MODELID${MODEL_ID}"
filename="./out/$today/BFS_${MODE^^}_METHOD${METHOD}_MODEL${MODEL_TYPE}${model_id_part}_SUBSTEPS${SUBSTEPS}_GPU${GPU_ID}.out"

# Construct the command to be executed
# cmd="/usr/bin/env CUDA_VISIBLE_DEVICES=$GPU_ID python Train_BFS.py \

cmd="/usr/bin/env python ./scripts/Train_BFS.py \
    --gpu_id $GPU_ID \
    --mode $MODE \
    --method $METHOD \
    --model_type $MODEL_TYPE \
    --substeps $SUBSTEPS"

# Add model_id to the command only if it's set
[ -n "$MODEL_ID" ] && cmd+=" --model_id $MODEL_ID"

# Execute the command in the background using nohup
nohup $cmd > "$filename" 2>&1 &

# --- Output Job Information ---
echo "Job started with the following settings:"
echo "  Mode:         $MODE"
echo "  Method:       $METHOD"
echo "  Model type:   $MODEL_TYPE"
echo "  Substeps:     $SUBSTEPS"
[ -n "$MODEL_ID" ] && echo "  Model ID:     $MODEL_ID"
echo "  GPU:          $GPU_ID"
echo "  ----------------------------"
echo "  Output file:  $filename"
echo "  PID:          $!"