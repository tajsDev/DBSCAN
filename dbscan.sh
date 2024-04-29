#!/bin/bash
#SBATCH --job-name=DBSCAN  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/ajm2327/dbscan.out #this is the file for stdout 
#SBATCH --error=/scratch/ajm2327/dbscan.err #this is the file for stderr

#SBATCH --time=24:00:00		#Job timelimit is 1 day
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24								

#compute capability
CC=70

module load cuda

nvcc -O3 -arch=compute_$CC -lcuda -lineinfo -Xcompiler -fopenmp dbscan.cu -o dbscan

srun ./dbscan 1199 2 1.0 5 sorted_smiley.csv
