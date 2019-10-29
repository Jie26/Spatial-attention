#!/bin/bash
#SBATCH --job-name='JW12_1'
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=50GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python track12_1.py

