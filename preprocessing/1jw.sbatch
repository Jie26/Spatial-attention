#!/bin/bash
#SBATCH --job-name='JW_enrich tracksdata'
#SBATCH --cpus=2
#SBATCH --gres=gpu:0
#SBATCH --mem=45GB
#SBATCH --time=3:00:00
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=END

# cd /usr/wiss/<your login>/<project_dir>/
python 1_enrichtracksdata.py

