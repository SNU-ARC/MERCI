# 4. Performance Evaluation

```bash
$ make all embedding_dim=${embeding_dimension}
```
## Eval MERCI

* `./bin/eval_merci --dataset <dataset name> --num_partition <# of partitions> --memory_ratio <size of memoization table> -c <# of threads> -r <# of repeats>`

## Eval Baseline

* `./bin/eval_baseline --dataset <dataset name> -c <# of threads> -r <# of repeats>`


## Eval Remapped

* `./bin/eval_remapped_only --dataset <dataset name> --num_partition <#of partitions> -c <# of threads> -r <# of repeats>`
