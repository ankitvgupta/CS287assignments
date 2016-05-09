#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=10000                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 20:00:00              #Indicate duration using HH:MM:SS
#SBATCH -p gpu             #Partition to submit to
#SBATCH --gres=gpu:1

#SBATCH -o /n/home11/tsilver/CS287/CS287assignments/finalproject/odyssey/outputs/out/rnn_bidirectional_memm_exp1/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home11/tsilver/CS287/CS287assignments/finalproject/odyssey/outputs/err/rnn_bidirectional_memm_exp1/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=tsilver@college.harvard.edu  #Email to which notifications will be sent

#!/bin/bash

# Read the config files and get the appropriate config
readarray -t config < /n/home11/tsilver/CS287/CS287assignments/finalproject/odyssey/scripts/rnn_bidirectional_memm_exp1_config.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

# Extract params
datafile=${selected_config[0]}
classifier=${selected_config[1]}
b=${selected_config[2]}
alpha=${selected_config[3]}
sequence_length=${selected_config[4]}
embedding_size=${selected_config[5]}
optimizer=${selected_config[6]}
epochs=${selected_config[7]}
hidden=${selected_config[8]}
eta=${selected_config[9]}
rnn1=${selected_config[10]}
rnn2=${selected_config[11]}
dropout=${selected_config[12]}
layers=${selected_config[13]}


# Run the trainer
cd /scratch
source /n/home11/tsilver/torch_setup.sh
th /n/home11/tsilver/CS287/CS287assignments/finalproject/finalproject.lua \
 -datafile /n/home11/tsilver/CS287/CS287assignments/finalproject/$datafile \
 -classifier $classifier \
 -b $b \
 -alpha $alpha \
 -sequence_length $sequence_length \
 -embedding_size $embedding_size \
 -optimizer $optimizer \
 -epochs $epochs \
 -hidden $hidden \
 -eta $eta \
 -rnn_unit1 $rnn1 \
 -rnn_unit2 $rnn2 \
 -dropout $dropout \
 -bidirectional_layers $layers \
 -odyssey \
 -bidirectional \
 -additional_features \
 -memm_layer \
 -cuda





