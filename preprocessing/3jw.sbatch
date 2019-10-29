#!/bin/bash
#SBATCH --job-name='JW'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=30GB
#SBATCH --time=10:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python 3_driving_scenario_rebuild.py

