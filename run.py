
import subprocess

def run_command_with_retries(command, max_retries=0):
    for attempt in range(max_retries):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
            else:
                print("All attempts failed, moving to the next command.")

datasets=['CiteSeer'] # Cora

for dataset in datasets:

    command = [
        "python", "Train_fedlag.py",
        "num_clients", str(dataset),
        "--dataset", str(dataset),
    ]
    print(f"Running command: {' '.join(command)}")
    run_command_with_retries(command, max_retries=1)

    command = [
        "python", "Train_fedtad.py",
        "--dataset", str(dataset),
    ]

    print(f"Running command: {' '.join(command)}")
    run_command_with_retries(command, max_retries=1)
