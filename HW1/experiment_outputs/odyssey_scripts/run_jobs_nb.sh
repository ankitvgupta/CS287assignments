#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=2000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 0:20:00              #Indicate duration using HH:MM:SS
#SBATCH -p serial_requeue               #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/HW1/outputs/out/exp2/setup_%A_%a.out            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/HW1/outputs/err/exp2/setup_%A_%a.err             #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

# Read the config files and get the appropriate config
readarray -t config < ~/CS287/HW1/scripts/config_nb.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

# Print the config variables
datafile=${selected_config[0]}
classifier=${selected_config[1]}
eta=${selected_config[2]}
lambda=${selected_config[3]}
minibatch=${selected_config[4]}
epochs=${selected_config[5]}
minlength=${selected_config[6]}
alpha=${selected_config[7]}
gen_validation_set=${selected_config[8]}

cd /scratch
source new-modules.sh
module load LuaJIT
module load luarocks
module load gcc/4.8.2-fasrc01 openmpi/1.10.1-fasrc01 hdf5/1.8.16-fasrc01
source /n/home09/ankitgupta/torch/install/bin/torch-activate

th /n/home09/ankitgupta/CS287/HW1/HW1.lua \
 -datafile /n/home09/ankitgupta/CS287/HW1/$datafile \
 -classifier $classifier \
 -eta $eta \
 -lambda $lambda \
 -minibatch $minibatch \
 -epochs $epochs \
 -min_sentence_length $minlength \
 -alpha $alpha \
 -generate_validation_set $gen_validation_set


