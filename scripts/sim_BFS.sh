#!/bin/bash
# chmod +x sim_BFS.sh
today=$(date +%Y-%m-%d)
mkdir -p "./out/$today"
sNrs=(0.9432)
ReNrs=(1290)
num_gpus=8  # Set this to the number of GPUs available
gpu_id=7 # Initial GPU ID
scale=${res_scale_x}x${res_scale_y}
for Re in "${ReNrs[@]}"; do
    for s in "${sNrs[@]}"; do
        filename="./out/$today/sim_BFS_Re${Re}_S${s}_GPU${gpu_id}.out"
        # Run the Python script and redirect output to a file
        nohup python Sim_BFS.py --s $s --Re $Re --gpu_id $gpu_id > "$filename" 2>&1 &

        # Output information about the started process
        echo "Simulation at s=$s, of Re$Re, started on GPU $gpu_id with PID $!"
        echo "Logging to $filename"

        # Increment GPU ID and wrap around if necessary
        ((gpu_id=(gpu_id+1)%num_gpus))
    done
done