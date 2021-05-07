# MERCI Code Repository

This repository contains all the codes and scripts written for the MERCI project (ASPLOS '21) @ [SNU Architecture and Code Optimization (ARC) Lab](http://arc.snu.ac.kr), 2021.

Please refer to the full paper at https://snu-arc.github.io/pubs/asplos21_merci.pdf.
<br />

### Dataset
Make sure that data is present at `$HOME/MERCI/1_raw_data` directory.
#### Amazon
Download reviews/metadata dataset from https://jmcauley.ucsd.edu/data/amazon/ to `$HOME/MERCI/1_raw_data/amazon`

e.g., `$HOME/MERCI/1_raw_data/amazon/meta_Office_Products.json.gz,Office_Products.json.gz` 
##### DBLP
Download dblp datset from http://networkrepository.com/ca-coauthors-dblp.php to `$HOME/MERCI/1_raw_data/dblp`
##### Lastfm
Download lastm dataset from http://millionsongdataset.com/lastfm/#getting to `$HOME/MERCI/1_raw_data/lastfm`

**control_dir_path.sh** generates data directory as shown below.
```bash
$ ./control_dir_path.sh ${dataset} ${num_partition}
# e.g., ./control_dir_path amazon_Office_Products 2748
```

```bash
$HOME/MERCI/1_raw_data
$HOME/MERCI/2_transactions/$dataset
$HOME/MERCI/3_train_test/$dataset
$HOME/MERCI/4_filtered/$dataset
$HOME/MERCI/5_patoh/$dataset/partition_$num_partition
$HOME/MERCI/6_evaluation_input/$dataset/partition_$num_partition
```

### 1. Preprocess
Process raw data into transactions, train/test sets, and filter them out accordingly
```bash
$ cd scripts
# Amazon
$ python3 amazon_parse_divide_filter.py Office_Products
# Other dataset
$ ./lastfm_dblp.sh dblp
```

### 2. Partition
Partition train dataset with PaToH algorithm
```bash
# Put latest PATOH binary in bin/
$ cd scripts
$ ./run_patoh.sh ${dataset} ${num_partition}
# e.g., ./run_patoh.sh amazon_Office_Products 2748
```

### 3. Clustering
Make sure PARTITION_SIZE in `clustering.cc` is set to Max value of PaToH result
```bash
'Con - 1' Cost: 1904732
Part Weights   : Min=        126 (0.007) Max=        128 (0.009)
```
```bash
$ mkdir bin
$ make
# Correlation-Aware Variable-Sized Clustering
$ ./bin/clustering -d ${dataset} -p ${num_partition}
# For Remapped in paper result
$ ./bin/clustering -d ${dataset} -p ${num_partition} --remap-only
```

### 4. Performance_Evaluation
Make sure PARTITION_SIZE is set to PARTITION_SIZE in clustering.cc 
BUF_SIZE in eval_merci.cc should be set to 1024 in case of dblp dataset

```bash
$ mkdir bin
$ make all
# eval baseline
$ ./bin/eval_baseline -d ${dataset} -r ${repeat} -b ${batch size} -c ${thread}
# eval merci
$ ./bin/eval_merci -d ${dataset} -p ${num_partition}  --memory_ratio ${mem} -c ${thread} -r ${repeat}
# eval remap only
$ ./bin/eval_remap_only -d ${dataset} -p ${num_partition} -c ${thread} -r ${repeat}
```

### Run all at once (e.g., Amazon Office Products, dblp)
```bash
$ ./run_all.sh
```

### Reproducing result in the paper
To reproduce the results in the paper, we recommend you to set up an instance on Amazon Web Services (AWS) EC2.
> m5.8xlarge instance (16 Intel Xeon Platinum 8259CL CPU cores with 128GiB of DRAM)
