#!/bin/bash
#SBATCH -p 3090-gcondo
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J precompute_data
#SBATCH -o logs/precompute_%j.out
#SBATCH -e logs/precompute_%j.err

# 1. Load Environment
module load anaconda/2023.09-0
module load cuda/11.8.0
source activate /users/erluo/scratch/faceswap_env

# --- CONFIGURATION (EDIT THESE PATHS) ---

# Input: Where your raw images are (e.g., from the download script)
INPUT_IMAGE_DIR="/users/erluo/scratch/faceswap-diffusion/ff-celeba-hq-dataset512"

# Output 1: Where to save the captions JSON
CAPTIONS_JSON="/users/erluo/scratch/ff-celba-hq-captions_detailed.json"

# Output 2: Where to save the temporary .pt embeddings
EMBEDDINGS_DIR="/users/erluo/scratch/ff-celeba-hq-embeddings"

# Output 3: The FINAL training dataset folder
FINAL_DATASET_DIR="/users/erluo/scratch/ff-celeba-hq-hrnet_dataset"

# Model Checkpoint (Path to glint360k_r100.pth)
FACEID_MODEL_PATH="/users/erluo/scratch/faceswap-diffusion/checkpoints/glint360k_r100.pth"

# ----------------------------------------

echo "Starting Data Precomputation Pipeline..."
echo "Input: $INPUT_IMAGE_DIR"

# STEP 1: Generate Captions (Qwen-VL)
# Skip this if you already have a captions.json
if [ ! -f "$CAPTIONS_JSON" ]; then
    echo "--- Step 1: Generating Captions ---"
    python scripts/dataset/generate_image_captions_qwen.py \
        --input_dir "$INPUT_IMAGE_DIR" \
        --output_json "$CAPTIONS_JSON" \
        --batch_size 8 \
        --style "detailed"
else
    echo "--- Step 1: Captions found ($CAPTIONS_JSON), skipping... ---"
fi

# STEP 2: Extract Face Embeddings (iResNet100)
# This creates a mirror folder structure with .pt files
echo "--- Step 2: Extracting Face Embeddings ---"
python scripts/dataset/iresnet_embedding_extraction.py \
    --input_dir "$INPUT_IMAGE_DIR" \
    --output_dir "$EMBEDDINGS_DIR" \
    --model_path "$FACEID_MODEL_PATH"

# STEP 3: Extract Landmarks & Assemble Dataset (HRNet)
# This combines images, embeddings, and landmarks into the final training format
echo "--- Step 3: Extracting Landmarks & assembling Final Dataset ---"
python scripts/dataset/prepare_hrnet_dataset.py \
    --input_dir "$INPUT_IMAGE_DIR" \
    --output_dir "$FINAL_DATASET_DIR" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --captions_json "$CAPTIONS_JSON" \
    --faceswap  # Enable FaceSwap tuple logic (Source/Target pairs)

echo "Done! Final dataset is at: $FINAL_DATASET_DIR"