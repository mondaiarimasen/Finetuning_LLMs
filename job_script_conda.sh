#! /bin/bash
#SBATCH --partition=gpu
#SBATCH -c4
#SBATCH --constraint=48g

# This line sets up Conda for shell interaction
eval "$($HOME/mc3/bin/conda 'shell.bash' 'hook')"

# Activate your conda environment
conda activate vzhangcondavenv
python "$1"