#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=5000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 10:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p general               #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/CS287assignments/HW2/odyssey/outputs/out/feb16/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/CS287assignments/HW2/odyssey/outputs/err/feb16/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

# Read the config files and get the appropriate config
readarray -t config < /n/home09/ankitgupta/CS287/CS287assignments/HW2/odyssey/scripts/config_nnet_adagrad.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

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

cd /scratch
source new-modules.sh
module load LuaJIT
module load luarocks
module load gcc/4.8.2-fasrc01 openmpi/1.10.1-fasrc01 hdf5/1.8.16-fasrc01
source /n/home09/ankitgupta/torch/install/bin/torch-activate
th /n/home09/ankitgupta/CS287/CS287assignments/HW2/HW2.lua \
  -datafile /n/home09/ankitgupta/CS287/CS287assignments/HW2/$datafile \
  -classifier $classifier \
  -eta $eta \
  -minibatch $minibatch \
  -epochs $epochs \
  -alpha $alpha \
  -lambda $lambda \
  -optimizer $optimizer \
  -hiddenlayers $hiddenlayers \
  -embedding_size $embeddingsize

