# List of label corruption probabilities to try
probabilities=(0.0 1.0)

# Python script to run experiments
script_name="train.py"

# Number of repetitions
num_repetitions=10

# GPU to use (3 three devices; 0, 1, or 2).
gpu_to_use=2  # Change this to 1 or 2 if you want to use a different GPU

# Printing the GPU they used.
echo "Using GPU $gpu_to_use"

# Outer loop to r/mnt/sdb/joel/playground/fitting-random-labels/runs/cifar10_corrupt1_wide-resnet28-1_lr0.1_mmt0.9_Wd0.0001_NoAug_28-11-24-18-30-17epeat the experiments
for ((rep=1; rep<=num_repetitions; rep++)); do
    echo "Starting repetition $rep of $num_repetitions"

    # Iterate over the list of probabilities
    for prob in "${probabilities[@]}"; do
        echo "Running experiment with label corruption probability: $prob (Repetition $rep)"

        # Set the GPU to use and run the Python script
        CUDA_VISIBLE_DEVICES="$gpu_to_use" python "$script_name" --label-corrupt-prob "$prob"

        echo "Experiment with label corruption probability $prob (Repetition $rep) completed."
    done

    echo "Repetition $rep of $num_repetitions completed."
done

echo "All experiments completed."
