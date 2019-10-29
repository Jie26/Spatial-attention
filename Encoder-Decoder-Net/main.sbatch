#!/bin/bash
#SBATCH --job-name='main'
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python main.py

