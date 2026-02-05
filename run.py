
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

dataset_scripts = {
    "CiteSeer": "fedlag_CiteSeer_1%_labels.py",
    "Cora": "fedlag_Cora_1%_labels.py",
}

datasets = ["CiteSeer", "Cora"]

for dataset in datasets:
    script = dataset_scripts.get(dataset)
    if script is None:
        print(f"Skipping unknown dataset: {dataset}")
        continue

    command = [
        "python", script,
        "--dataset", str(dataset),
    ]
    print(f"Running command: {' '.join(command)}")
    run_command_with_retries(command, max_retries=1)
