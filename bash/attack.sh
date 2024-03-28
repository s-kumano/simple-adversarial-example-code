#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}

img_names=(cat.JPG plate.JPG)
perturbation_sizes=(4 8 16 24 32 64)
img_sizes=(224 400)

for img_name in "${img_names[@]}"; do
for perturbation_size in "${perturbation_sizes[@]}"; do
for img_size in "${img_sizes[@]}"; do
  python3 py/attack.py $img_name $perturbation_size -i $img_size -d $devices >> logs/${now}.out 2>&1
  python3 py/attack.py $img_name $perturbation_size -t 0 -i $img_size -d $devices >> logs/${now}.out 2>&1
done
done
done