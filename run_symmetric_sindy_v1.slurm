#!/bin/bash
#SBATCH --job-name=symm_sindy_v1
#SBATCH --partition=ckpt-all
#SBATCH --account=stf
#SBATCH --gres=gpu:1
#SBATCH --mail-user=songk42@uw.edu
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --chdir=/mmfs1/home/songk42/ehd-sindy
#SBATCH --output=/mmfs1/home/songk42/ehd-sindy/logs/%x_%j.log
#SBATCH --time=12:00:00

module load cuda/12.9.1
module load gcc/11.2.0


cd /mmfs1/home/songk42/ehd-sindy
source .venv/bin/activate
python notebooks/symmetric_sindy_v1.py