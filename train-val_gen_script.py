# generating sh file for training specified file for specified number of epochs

epochs = 10

sh_file = 'train-val_script.sh'
run_file = "training_loop_w_checkpoints.py"

with open(sh_file, 'w') as file:
    file.write("#!/bin/bash\n\n")

for i in range(epochs):
    with open(sh_file, 'a') as file:
        file.write(f"sbatch -J tl-ft-gpt2 -d singleton job_script_conda.sh {run_file}\n")




