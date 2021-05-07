# $1: dataset name (e.g., dblp, Office_Products)
# $2: number of partition (e.g., 3300, 2748)
basedir=$HOME/MERCI/data

mkdir -p $basedir/4_filtered/$1
mkdir -p $basedir/5_patoh/$1/partition_$2
mkdir -p $basedir/6_evaluation_input/$1/partition_$2
