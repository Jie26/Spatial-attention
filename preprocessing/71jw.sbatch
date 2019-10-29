#!/bin/bash
#SBATCH --job-name='JW71'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:0
#SBATCH --mem=50GB
#SBATCH --time=5:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python 71track_sequence.py

