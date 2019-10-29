#!/bin/bash
#SBATCH --job-name='JW41'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=5GB
#SBATCH --time=10:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python 41label+scenario.py

