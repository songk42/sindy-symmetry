#!/bin/bash
# Full dysts rerun (all sections of symmetric_sindy_dysts.md) on klone, into logs/dysts_traj/.
set -e
cd /mmfs1/home/songk42/ehd-sindy
mkdir -p logs/dysts_traj/slurm
S=run_dysts_klone.slurm
si=$(sbatch --parsable -J sindy_si $S sindy_si_dysts.py)
echo "sindy_si $si"
for a in 0 0.05; do for nrm in 0 1; do
  sbatch -J stlsq_a${a}_n${nrm} --export=ALL,SINDY_ALPHA=$a,SINDY_NORMALIZE=$nrm $S symmetric_sindy_dysts.py
done; done
for nrm in 0 1; do
  sbatch -J sr3_n${nrm} --export=ALL,SINDY_METHODS=sr3,SINDY_NORMALIZE=$nrm $S symmetric_sindy_dysts.py
done
for side in eq eq_mono; do
  for a in 0 0.05; do for nrm in 0 1; do
    sbatch -J stlsq_a${a}_n${nrm}_$side -d afterok:$si --export=ALL,SINDY_ALPHA=$a,SINDY_NORMALIZE=$nrm,SINDY_SIDE=$side $S symmetric_sindy_dysts.py
  done; done
  for nrm in 0 1; do
    sbatch -J sr3_n${nrm}_$side -d afterok:$si --export=ALL,SINDY_METHODS=sr3,SINDY_NORMALIZE=$nrm,SINDY_SIDE=$side $S symmetric_sindy_dysts.py
  done
done
sbatch -J data_eff -d afterok:$si --array=0-41 -c 2 --mem=4G --time=4:00:00 $S data_efficiency_dysts.py
