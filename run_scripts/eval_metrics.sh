#!/bin/bash

python scripts/evaluation/run_all_metrics.py \
    --source_dir test_set/sources \
    --target_dir test_set/targets \
    --swapped_dir swapped_outputs \
    --pairs_json test_set/pairs.json \
    --output results/evaluation_results.json \
    --gpu 0

echo "End to end metric eval completed"