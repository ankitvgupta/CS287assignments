#!/bin/bash

#SBATCH -n 1                         #Number of cores
#SBATCH -N 1                          #Run on 1 node
#SBATCH --mem=1500                  #Memory per cpu in MB (see also --mem)

#SBATCH -t 3:30:00              #Indicate duration using HH:MM:SS
#SBATCH -p general               #Partition to submit to

#SBATCH -o /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/outputs/out/hmm_exp1/setup_%A_%a_out.txt            #File to which standard out will be written
#SBATCH -e /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/outputs/err/hmm_exp1/setup_%A_%a_err.txt            #File to which standard err will be written
#SBATCH --mail-type=ALL                 #Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ankitgupta@college.harvard.edu  #Email to which notifications will be sent

# Read the config files and get the appropriate config
readarray -t config < /n/home09/ankitgupta/CS287/CS287assignments/HW5/odyssey/scripts/hmm_experiment1_config.txt
selected_config=(${config[$SLURM_ARRAY_TASK_ID]})

# Extract params
datafile=${selected_config[0]}
classifier=${selected_config[1]}
window_size=${selected_config[2]}
b=${selected_config[3]}
alpha=${selected_config[4]}
sequence_length=${selected_config[5]}
embedding_size=${selected_config[6]}
optimizer=${selected_config[7]}
epochs=${selected_config[8]}
hidden=${selected_config[9]}
eta=${selected_config[10]}

# Run the trainer
cd /scratch
source /n/home09/ankitgupta/torch_setup.sh
th /n/home09/ankitgupta/CS287/CS287assignments/HW5/HW5.lua \
  -datafile /n/home09/ankitgupta/CS287/CS287assignments/HW5/$datafile \
  -classifier $classifier \
  -alpha $alpha \
  -beta $beta \
  -embedding_size $embedding_size \
  - minibatch_size $minibatch_size \
  -optimize $optimizer \
  -epochs $epochs \
  -hidden $hidden \
  -eta $eta \
  -odyssey



  cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use')
cmd:option('-alpha', 1, 'laplacian smoothing factor')
cmd:option('-beta', 1, 'F score parameter')
cmd:option('-odyssey', false, 'Set to true if running on odyssey')
cmd:option('-testfile', '', 'test file (must be HDF5)')
cmd:option('-embedding_size', 50, 'Size of embeddings')
cmd:option('-minibatch_size', 320, 'Size of minibatches')
cmd:option('-optimizer', 'sgd', 'optimizer to use')
cmd:option('-epochs', 10, 'Number of epochs')
cmd:option('-hidden', 50, 'Hidden layer (for nn only)')
cmd:option('-eta', 1, 'Learning rate')


