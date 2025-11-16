#!/bin/bash
# command example: ./train_KS.sh train 0
# command example with model_id: ./train_KS.sh train 0 model_id 0
# --------------------------------------------
declare -a MODEL_TYPES_CORR=("FNO" "UNet" "DeepONet" "ResNet")
CORR_TERM="INC"

# Default settings
DEFAULT_GPU=0
TASK="KS"
DEFAULT_down_ratio=8
# MSTEP=5
# test_step=50
# dt=5e-1

MSTEP=50
test_step=5000
dt=1e-2
# --------------------------------------------

show_usage() {
    echo "Usage:"
    echo "  Training:"
    echo "    $0 train [model_type_idx] [model_id (optional)] [gpu_id (optional)]"
    echo "  Testing:"
    echo "    $0 test  [model_type_idx] [model_id] [gpu_id (optional)]"
    echo ""
    echo "Allowed parameters:"
    echo "  Model types: ${!MODEL_TYPES_CORR[@]} -> (${MODEL_TYPES_CORR[@]})"
    exit 1
}

run_experiment() {
    local python_script=$1
    shift
    local gpu=$1
    shift
    
    # GPU is now handled by the Python script through --gpu_id parameter
    # nohup /usr/bin/env python $python_script "$@" > /dev/null 2>&1 &
    nohup /usr/bin/env python $python_script "$@" > nohup_KS.out 2>&1 &
    local pid=$!
    echo "Job started with PID $pid (GPU $gpu)"
    disown $pid  # Remove job from shell's job table
}

run_correction() {
    local mode=$1
    shift

    if [ "$mode" == "train" ]; then
        local mt_idx=$1
        # Handle optional model_id and gpu
        local model_id=""
        local gpu=$DEFAULT_GPU
        if [ $# -ge 2 ] && [[ ! $2 =~ ^[0-9]+$ ]]; then
            model_id=$2
            if [ $# -ge 3 ]; then
                gpu=${3}
            fi
        else
            if [ $# -ge 2 ]; then
                gpu=${2}
            fi
        fi
    elif [ "$mode" == "test" ]; then
        local mt_idx=$1
        local model_id=$2
        local gpu=${3:-$DEFAULT_GPU}
    else
        echo "Invalid mode: $mode"
        exit 1
    fi

    # Validate indices
    [ -z "${MODEL_TYPES_CORR[$mt_idx]}" ] && echo "Invalid model_type index" && exit 1

    local base_cmd="--mode $mode \
        --task $TASK \
        --dt $dt \
        --test_step $test_step \
        --down_ratio $DEFAULT_down_ratio \
        --model_type ${MODEL_TYPES_CORR[$mt_idx]} \
        --correction_term $CORR_TERM \
        --mstep $MSTEP \
        --gpu_id $gpu"

    if [ "$mode" == "train" ]; then
        if [ -n "$model_id" ]; then
            base_cmd+=" --model_id $model_id"
        fi
        run_experiment ./scripts/Train_1D.py  $gpu $base_cmd
    else
        [ -z "$model_id" ] && echo "Missing model_id for test mode" && exit 1
        run_experiment ./scripts/Train_1D.py $gpu $base_cmd --model_id "$model_id"
    fi
}

# Main execution
[ $# -lt 2 ] && show_usage

mode=$1
shift

case $mode in
    "train")
        [ $# -lt 1 ] && show_usage
        run_correction train $1 $2 $3
        ;;
    "test")
        [ $# -lt 2 ] && show_usage
        run_correction test $1 $2 $3
        ;;
    *)
        show_usage
        ;;
esac