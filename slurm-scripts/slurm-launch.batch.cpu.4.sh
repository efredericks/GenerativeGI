#!/bin/bash

#SBATCH --mem=6144
#SBATCH --time=64:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fredericks-gengi-args-ec4 ## Name of job
#SBATCH --output=fredericks-gengi-args-ec4.%j.%A.%a.out ##Name of output file
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=frederer@gvsu.edu

# Array things
#SBATCH --array 1-15

d="/mnt/home/frederer/GenerativeGI-withargs"
w="ec4"
source /mnt/home/frederer/gengi/bin/activate
#srun time python3 $d/deap_main.py --gens 100 --pop_size 100 --treatment ${SLURM_ARRAY_TASK_ID} --run_num ${SLURM_ARRAY_TASK_ID} --output_path $d/$w/${SLURM_ARRAY_TASK_ID} --lexicase --shuffle --ff_rms --ff_gc --ff_ut --ff_cheby --ff_neg --ff_art

#srun time python3 $d/deap_main.py --gens 100 --pop_size 100 --treatment ${SLURM_ARRAY_TASK_ID} --run_num ${SLURM_ARRAY_TASK_ID} --output_path $d/$w/${SLURM_ARRAY_TASK_ID} --lexicase --shuffle --ff_rms --ff_gc --ff_ut --ff_cheby
srun time python3 $d/deap_main.py --gens 100 --pop_size 100 --treatment ${SLURM_ARRAY_TASK_ID} --run_num ${SLURM_ARRAY_TASK_ID} --output_path $d/$w/${SLURM_ARRAY_TASK_ID} --lexicase --shuffle --ff_rms --ff_gc --ff_ut --ff_cheby
