PRECOMPUTED_DIR="/users/erluo/scratch/ff-celeba-hq-hrnet_dataset"
EVAL_ROOT="/users/erluo/scratch/faceswap-diffusion/evaluation/test_dataset_detailed_reloaded"

python /users/erluo/scratch/faceswap-diffusion/scripts/dataset/prepare_test_dataset.py \
    --precomputed_dir "$PRECOMPUTED_DIR" \
    --output_dir "$EVAL_ROOT" \
    --num_samples 1000