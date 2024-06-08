#!/bin/bash

# #SBATCH --nodes=5 ##Number of nodes I want to use
# #SBATCH --mem=6144# 4096 ##Memory I want to use in MB
# #SBATCH --time=64:00:00 # 48:00:00 ## time it will take to complete job
# #SBATCH --partition=cpu # all ##Partition I want to use
# #SBATCH --ntasks=15 ##Number of task
# #SBATCH --cpus-per-task=4 # 8
# #SBATCH --job-name=fredericks-gengi-args-ec9 ## Name of job
# #SBATCH --output=fredericks-gengi-args-ec9.%j.%A.%a.out ##Name of output file
# #SBATCH --mail-type=END,FAIL,TIME_LIMIT
# #SBATCH --mail-user=frederer@gvsu.edu

#SBATCH --mem=6144
#SBATCH --time=64:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fredericks-gengi-args-ec8 ## Name of job
#SBATCH --output=fredericks-gengi-args-ec8.%j.%A.%a.out ##Name of output file
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=frederer@gvsu.edu

# Array things
#SBATCH --array 1-15

#d="/home/frederer/GenerativeGI"
d="/mnt/home/frederer/GenerativeGI-withargs"
w="ec9"
source /mnt/home/frederer/gengi/bin/activate
#source /home/frederer/gengi/bin/activate
#time python3 $d/deap_main.py --gens 100 --pop_size 100 --treatment ${SLURM_ARRAY_TASK_ID} --run_num ${SLURM_ARRAY_TASK_ID} --output_path $d/$w/${SLURM_ARRAY_TASK_ID} --lexicase --shuffle --ff_rms --ff_gc --ff_ut --ff_cheby --ff_neg --ff_art
srun time python3 $d/deap_main.py --gens 100 --pop_size 100 --treatment ${SLURM_ARRAY_TASK_ID} --run_num ${SLURM_ARRAY_TASK_ID} --output_path $d/$w/${SLURM_ARRAY_TASK_ID} --lexicase --shuffle --ff_rms --ff_gc --ff_cheby --ff_neg 
