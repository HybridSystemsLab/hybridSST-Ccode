#!/bin/bash

echo "Enter desired solution batch size (1-7): "
read batch_size
while ! [[ "$batch_size" =~ ^[1-7]$ ]]; do
    echo "Batch size must be an integer between 1 and 7. Please try again: "
    read batch_size
done
while ! [[ "$iterations" =~ ^[0-9]+$ ]]; do
    echo "Enter desired number of iterations: "
    read iterations
done

for ((i=1; i<=$iterations; i++)); do
    if [ "$batch_size" -eq 1 ]; then
        ./run1
    elif [ "$batch_size" -eq 2 ]; then
        ./run2
    elif [ "$batch_size" -eq 3 ]; then
        ./run3
    elif [ "$batch_size" -eq 4 ]; then
        ./run4
    elif [ "$batch_size" -eq 5 ]; then
        ./run5
    elif [ "$batch_size" -eq 6 ]; then
        ./run6
    elif [ "$batch_size" -eq 7 ]; then
        ./run7
    fi
done