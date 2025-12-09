RAW_DATASET="/users/erluo/scratch/faceswap-diffusion/ffhq-dataset512"
PRECOMPUTED_DIR="/users/erluo/scratch/ff-celeba-hq-hrnet_dataset"
CAPTIONS_FILE="/users/erluo/scratch/captions.jsonl"
EVAL_ROOT="/users/erluo/scratch/faceswap-diffusion/evaluation/test_dataset_detailed"

python /users/erluo/scratch/faceswap-diffusion/scripts/dataset/prepare_test_dataset.py \
    --dataset_dir "$RAW_DATASET" \
    --precomputed_dir "$PRECOMPUTED_DIR" \
    --captions_path "$CAPTIONS_FILE" \
    --output_dir "$EVAL_ROOT" \
    --num_samples 1000