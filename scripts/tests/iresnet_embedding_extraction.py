import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from scripts.models.iresnet import iresnet100
from PIL import Image
import torchvision.transforms as T
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "glint360k_r100.pth"

model = iresnet100(pretrained=False, fp16=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 2. Preprocessing (Standard for ArcFace)
transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(img_tensor)
    
    return F.normalize(emb, p=2, dim=1)

img_personA_1 = "trump1.png" 
img_personA_2 = "trump2.png" # Same person, different photo
img_personB_1 = "00011.png" # Different person

if __name__ == "__main__":
    try:
        emb1 = get_embedding(img_personA_1)
        # print(f"emb1 shape: {emb1.shape}")
        emb2 = get_embedding(img_personA_2)
        emb3 = get_embedding(img_personB_1)

        sim_same = torch.sum(emb1 * emb2).item()
        sim_diff = torch.sum(emb1 * emb3).item()

        print(f"--- Results for Glint360K ResNet100 (without detection) ---")
        print(f"Same Person Similarity:      {sim_same:.4f}")
        print(f"Different Person Similarity: {sim_diff:.4f}")
        
        if sim_same > 0.3 and sim_diff < 0.1:
            print("\nGOOD. Model works.")
        else:
            print("\nBAD. Values less than expected.")
            
    except Exception as e:
        print(f"Error: {e}")