import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FaceAnalysis(
    name="antelopev2", # will still expect /models/antelopev2/
    root=str(Path(__file__).parent.parent.parent),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    faces = app.get(image)
    
    if len(faces) == 0:
        # Try with smaller detection size
        app.det_model.input_size = (512, 512)
        faces = app.get(image)
        app.det_model.input_size = (640, 640)
        
        if len(faces) == 0:
            raise ValueError(f"No faces detected in: {image_path}")
    
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    
    emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    # ============== DEBUG VISUALIZATION START =================
    # print(f"faces: {len(faces)}")
    # os.makedirs("debug_check", exist_ok=True)

    # # Draw the bounding box (Green)
    # bbox = faces[0].bbox.astype(int)
    # debug_img = cv2.imread(str(image_path)).copy()
    # cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # if faces[0].kps is not None:
    #     for kp in faces[0].kps:
    #         cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

    # # Save it to check visually
    # cv2.imwrite(f"debug_check/debug_{os.path.basename(image_path)}", debug_img)
    # print(f"face emb size: {emb.shape}")
    # ============== DEBUG VISUALIZATION END =================
    
    emb = F.normalize(emb, p=2, dim=1)
    
    return emb

img_personA_1 = "trump1.png" 
img_personA_2 = "trump2.png" # Same person, different photo
img_personB_1 = "00011.png" # Different person

if __name__ == "__main__":
    try:
        emb1 = get_embedding(img_personA_1)
        emb2 = get_embedding(img_personA_2)
        emb3 = get_embedding(img_personB_1)

        sim_same = torch.sum(emb1 * emb2).item()
        sim_diff = torch.sum(emb1 * emb3).item()

        print(f"--- Results for InsightFace AntelopeV2 (with detection) ---")
        print(f"Same Person Similarity:      {sim_same:.4f}")
        print(f"Different Person Similarity: {sim_diff:.4f}")
        
        if sim_same > 0.5 and sim_diff < 0.1:
            print("\nGOOD. Model works.")
        else:
            print("\nBAD. Values less than expected.")
            
    except Exception as e:
        print(f"Error: {e}")

