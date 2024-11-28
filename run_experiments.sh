# List of label corruption probabilities to try
probabilities=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Python script to run experiments
script_name="train.py"

# Number of repetitions
num_repetitions=10

# Outer loop to repeat the experiments
for ((rep=1; rep<=num_repetitions; rep++)); do
    echo "Starting repetition $rep of $num_repetitions"

    # Iterate over the list of probabilities
    for prob in "${probabilities[@]}"; do
        echo "Running experiment with label corruption probability: $prob (Repetition $rep)"

        # Run the Python script with the current corruption probability
        python "$script_name" --label-corrupt-prob "$prob"

        echo "Experiment with label corruption probability $prob (Repetition $rep) completed."
    done

    echo "Repetition $rep of $num_repetitions completed."
done

echo "All experiments completed."
