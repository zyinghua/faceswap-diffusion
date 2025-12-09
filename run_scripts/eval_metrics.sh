#!/bin/bash

python scripts/evaluation/run_all_metrics.py \
    --source_dir test_set/sources \
    --target_dir test_set/targets \
    --swapped_dir swapped_outputs \
    --pairs_json test_set/pairs.json \
    --output results/evaluation_results.json \
    --gpu 0

echo "End to end metric eval completed"


python scripts/evaluation/compute_id_similarity.py \
    /users/erluo/scratch/faceswap-diffusion/evaluation_imgs/test_dataset_detailed_reloaded/pairs.json \
    --source-dir evaluation_imgs/test_dataset_detailed_reloaded/sources \
    --swapped-dir evaluation_imgs/test_dataset_detailed_reloaded/swaps/images \
    --output /users/erluo/scratch/faceswap-diffusion/evaluation_imgs/test_dataset_detailed_reloaded/id_similarity_results.json


python scripts/evaluation/compute_id_retrieval.py \
    /users/erluo/scratch/faceswap-diffusion/evaluation_imgs/test_dataset_detailed_reloaded/pairs.json \
    /users/erluo/scratch/faceswap-diffusion/evaluation_imgs/test_dataset_detailed_reloaded/retrieval_db.json