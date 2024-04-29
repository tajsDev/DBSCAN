#!/bin/bash
#SBATCH --job-name=DBSCAN  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/ndr82/dbscan.out #this is the file for stdout 
#SBATCH --error=/scratch/ndr82/dbscan.err #this is the file for stderr

#SBATCH --time=24:00:00		#Job timelimit is 1 day
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C v100 #GPU Model: k80, p100, v100, a100
#SBATCH --account=cs453-spr24								

#compute capability
CC=70

module load cuda
make baseline
srun ./dbscan 5000000 2 100.0 5 sortedAST2.csv
