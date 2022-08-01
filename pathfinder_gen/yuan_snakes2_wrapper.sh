#!/bin/bash
n_machines=2
num_images=10
imgs_path='/home/AD/yutang/pathfinder21_dataset/'
script_name='/home/AD/yutang/pathfinder_data-main/snakes2_wrapper.py'

for i_machine in $(seq 1 $n_machines); do
python $script_name $n_machines $i_machine $num_images $imgs_path
done