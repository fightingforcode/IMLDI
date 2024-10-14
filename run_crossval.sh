#!/bin/bash

# Define the range of attention layers and heads
attention_layers=(4 5 6)
attention_heads=(8 12)
learning_rate=(1e-5 1e-6)
# Loop through each fold from 1 to 10
for fold in {1..5}
do
  # Loop through each attention layer
  for layer in "${attention_layers[@]}"
  do
    # Loop through each attention head
    for head in "${attention_heads[@]}"
    do
      for lr in "${learning_rate[@]}"
      do
        python train.py --fold-num $fold --mode 'cross_val' --depth $layer --num-heads $head --lr $lr >/dev/null 2>&1 &
        wait
      done
    done
  done
done
