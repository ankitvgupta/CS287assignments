#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=1500                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 3:30:00              #Indicate duration using HH:MM:SS
#SBATCH -p general               #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/outputs/out/struct_exp1/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/outputs/err/struct_exp1/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

# Read the config files and get the appropriate config
readarray -t config < /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/scripts/struct_experiment1_config.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

# Extract params
datafile=${selected_config[0]}
classifier=${selected_config[1]}
alpha=${selected_config[2]}
beta=${selected_config[3]}
embedding_size=${selected_config[4]}
minibatch_size=${selected_config[5]}
optimizer=${selected_config[6]}
epochs=${selected_config[7]}
hidden=${selected_config[8]}
eta=${selected_config[9]}

# Run the trainer
cd /scratch
source /n/home09/ankitgupta/torch_setup.sh
th /n/home09/ankitgupta/CS287/CS287assignments/HW5/HW5.lua \
  -datafile /n/home09/ankitgupta/CS287/CS287assignments/HW5/$datafile \
  -classifier $classifier \
  -alpha $alpha \
  -beta $beta \
  -embedding_size $embedding_size \
  -minibatch_size $minibatch_size \
  -optimizer $optimizer \
  -epochs $epochs \
  -hidden $hidden \
  -eta $eta \
  -odyssey





