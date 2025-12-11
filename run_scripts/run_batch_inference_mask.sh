EVAL_ROOT="/users/erluo/scratch/faceswap-diffusion/evaluation/test_dataset_detailed_reloaded"

echo "Starting Batch Inference..."
echo "Reading config from: $EVAL_ROOT/inference_config.json"
echo "Saving results to:   /users/erluo/scratch/faceswap-diffusion/evaluation_imgs/batch_inference"

python /users/erluo/scratch/faceswap-diffusion/scripts/inference/run_batch_inference_mask.py \
    --config_json "$EVAL_ROOT/inference_config.json" \
    --output_dir "/users/erluo/scratch/faceswap-diffusion/evaluation_imgs/batch_inference" \

echo "Done! Images generated."