#!/bin/bash
#SBATCH --job-name=DBSCAN  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/ajm2327/CPUdbscan.out #this is the file for stdout 
#SBATCH --error=/scratch/ajm2327/CPUdbscan.err #this is the file for stderr

#SBATCH --time=24:00:00    #Job timelimit is 1 day
#SBATCH --mem=30000        #memory requested in MiB
#SBATCH --account=cs453-spr24

# Load the GCC module (if available)
module load gcc

# Compile the C program using GCC
gcc -O3 -o dbscan dbscanCPU.c -lm

# Run the compiled program
srun ./dbscan