EVAL_ROOT="/users/erluo/scratch/faceswap-diffusion/evaluation/test_dataset_detailed"
CONTROLNET_PATH="/users/erluo/scratch/faceswap-diffusion/checkpoints/faceswap-model/checkpoint-30000/controlnet"
IP_ADAPTER_PATH="/users/erluo/scratch/faceswap-diffusion/checkpoints/faceswap-model/checkpoint-30000/ip_adapter/ip_adapter.bin"
FACEID_ENC_PATH="checkpoints/glint360k_r100.pth"

echo "Starting Batch Inference..."
echo "Reading config from: $EVAL_ROOT/inference_config.json"
echo "Saving results to:   $EVAL_ROOT/swaps"

python /users/erluo/scratch/faceswap-diffusion/scripts/inference/run_batch_inference.py \
    --config_json "$EVAL_ROOT/inference_config.json" \
    --output_dir "$EVAL_ROOT/swaps" \
    --controlnet_path "$CONTROLNET_PATH" \
    --ip_adapter_path "$IP_ADAPTER_PATH" \
    --faceid_encoder_path "$FACEID_ENC_PATH"

echo "Done! Images generated."