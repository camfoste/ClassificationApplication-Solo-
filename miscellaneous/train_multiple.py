import subprocess

learning_rates = [0.0005, 0.005, 0.05]

# Call train.py with the specified learning rate, fixed batch size and epochs
    # This automates multiple training runs for comparison and logging
for lr in learning_rates:
    subprocess.run([
        "python", "train.py",
        "--epochs", "50",
        "--lr", str(lr),
        "--batch_size", "64"
    ])
