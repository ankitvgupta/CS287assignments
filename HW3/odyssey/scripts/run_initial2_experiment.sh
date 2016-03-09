#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=4000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 15:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p serial_requeue               #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/CS287assignments/HW3/odyssey/outputs/out/nn/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/CS287assignments/HW3/odyssey/outputs/err/nn/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

# Read the config files and get the appropriate config
readarray -t config < /n/home09/ankitgupta/CS287/CS287assignments/HW3/odyssey/scripts/initial_experiment2_config.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

# Extract params
datafile=${selected_config[0]}
classifier=${selected_config[1]}
eta=${selected_config[2]}
minibatch=${selected_config[3]}
epochs=${selected_config[4]}
alpha=${selected_config[5]}
lambda=${selected_config[6]}
optimizer=${selected_config[7]}
hiddenlayers=${selected_config[8]}
embeddingsize=${selected_config[9]}

# Run the trainer
cd /scratch
source /n/home09/ankitgupta/torch_setup.sh
th /n/home09/ankitgupta/CS287/CS287assignments/HW3/HW3.lua \
  -datafile /n/home09/ankitgupta/CS287/CS287assignments/HW3/$datafile \
  -classifier $classifier \
  -eta $eta \
  -minibatch $minibatch \
  -epochs $epochs \
  -alpha $alpha \
  -lambda $lambda \
  -optimizer $optimizer \
  -hiddenlayers $hiddenlayers \
  -embedding_size $embeddingsize \
  -odyssey


