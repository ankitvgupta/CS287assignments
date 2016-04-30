#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=1000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 10:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p serial_requeue             #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/CS287assignments/finalproject/odyssey/outputs/out/filter_seq_1/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/CS287assignments/finalproject/odyssey/outputs/err/filter_seq_1/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

#!/bin/bash


# Run the trainer
cd /scratch
source /n/home09/ankitgupta/biopython_setup.sh
python /n/home09/ankitgupta/CS287/CS287assignments/finalproject/princfilter.py \
 /n/home09/ankitgupta/CS287/CS287assignments/finalproject/data/cb513+profile_split1.npy \
 /n/home09/ankitgupta/CS287/CS287assignments/finalproject/data/ss.txt \
 $(($SLURM_ARRAY_TASK_ID * 250)) \
 250


