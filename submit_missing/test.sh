#!/bin/bash
source missing.env

for combo in "${COMBINATIONS_GTN[@]}"; do
    read model dataset <<< "$combo"
    echo "GTN: $model $dataset"
done
