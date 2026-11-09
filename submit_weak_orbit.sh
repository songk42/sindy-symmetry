#!/bin/bash
# Weak form x orbit thresholding on klone (see notebooks/weak_orbit_dysts.py, weak_orbit_data_eff.py).
# One array task per (system, noise) = 42 x 4. Outputs in logs/dysts_traj/weak_orbit_{dysts,data_eff}/m25_p8/.
set -e
cd /mmfs1/home/songk42/ehd-sindy
S=run_dysts_klone.slurm
sbatch -J wo_full --array=0-167 -c 2 --mem=4G --time=3:00:00 $S weak_orbit_dysts.py
sbatch -J wo_de --array=0-167 -c 2 --mem=4G --time=10:00:00 $S weak_orbit_data_eff.py
