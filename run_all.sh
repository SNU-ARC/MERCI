#!/bin/bash

# Dataset
# DATASET=("amazon_Sports_and_Outdoors" "amazon_Clothing_Shoes_and_Jewelry" "amazon_Electronics" "amazon_Home_and_Kitchen" "amazon_Office_Products"
# "lastfm" "dblp" "amazon_Books")
# num_partitions=( 6340 7200 3720 8000 2748 3370 3300 18400 )
DATASET=("amazon_Office_Products" "dblp" )
num_partitions=( 2748  3300 )

# Variables
thread=32 # check hw core
len=${#DATASET[@]}
mem_sizes=( 1.25 1.5 2.0 9.0 )

# Output directories
top_dir=result
base_dir=$top_dir/baseline
remap_dir=$top_dir/remap_only
merci_dir=$top_dir/merci

mkdir -p $top_dir
mkdir -p $base_dir
mkdir -p $remap_dir
mkdir -p $merci_dir

# Set up data directories
for (( i=0; i<$len; i++ )); do
    ./control_dir_path.sh ${DATASET[$i]} ${num_partitions[$i]}
done

# 1. Preprocess
cd 1_preprocess/scripts
# Amazon Office_Products
python3 amazon_parse_divide_filter.py Office_Products
# dblp, lastfm
./lastfm_dblp.sh dblp

#2. Partition
cd ../../2_partition/scripts
for (( i=0; i<$len; i++ )); do
    ./run_patoh.sh ${DATASET[$i]} ${num_partitions[$i]}
done

#3. Clustering
cd ../../3_clustering
mkdir bin
make

#4. Performance Evaluation
cd ../4_performance_evaluation
mkdir bin
make all

# Baseline
for dataset in ${DATASET[@]}; do
    printf "\nRunning baseline on dataset %s\n" ${dataset}
    sync && echo 1 > /proc/sys/vm/drop_caches
    ./bin/eval_baseline -d ${dataset} -c ${thread} -r 5 > ../${base_dir}/${dataset}
done


# Remap only
for (( i=0; i<$len; i++ )); do
    printf "\nRunning remap-only on dataset %s\n" ${DATASET[$i]}
    ../3_clustering/bin/clustering -d ${DATASET[$i]} -p ${num_partitions[$i]} --remap-only
    sync && echo 1 > /proc/sys/vm/drop_caches
    ./bin/eval_remap_only -d ${DATASET[$i]} -c ${thread} -r 5 -p ${num_partitions[$i]} > ../${remap_dir}/${DATASET[$i]}
done


# MERCI (1.25x, 1.5x, 1x, 8x)
for (( i=0; i<$len; i++ )); do
    printf "\nRunning MERCI on dataset %s\n" ${DATASET[$i]}
   ../3_clustering/bin/clustering -d ${DATASET[$i]} -p ${num_partitions[$i]}
    for mem in ${mem_sizes[@]}; do
        sync && echo 1 > /proc/sys/vm/drop_caches
        ./bin/eval_merci -d ${DATASET[$i]} -p ${num_partitions[$i]}  --memory_ratio ${mem} -c ${thread} -r 5 > ../${merci_dir}/${DATASET[$i]}_${mem}X
    done
done
