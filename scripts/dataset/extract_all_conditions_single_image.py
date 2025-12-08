"""
Script for extracting all conditions {mask, Face ID embedding, landmark, caption} from a single image.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import facer
from PIL import Image
import mediapipe as mp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from insightface.app import FaceAnalysis

from scripts.models.iresnet import iresnet100

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
cache_dir = "/root/autodl-tmp"

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    cache_dir=cache_dir,
)
qwen_processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)


class FacialMaskExtractor:
    def __init__(self, dilated=False):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.dilated = dilated
        
        if dilated:
            self.FACE_OVAL = [
                10, 151, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401,
                172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
                397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338,
                18, 200, 199, 175,
                234, 127, 162, 21, 54, 103, 67, 109,
                172, 58, 132, 93
            ]
        else:
            self.FACE_OVAL = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
    
    def extract_mask(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        face_points = []
        h, w = image.shape[:2]
        for idx in self.FACE_OVAL:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_points.append([x, y])
        
        face_points = np.array(face_points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.dilated:
            hull = cv2.convexHull(face_points)
            cv2.fillPoly(mask, [hull], 255)
        else:
            cv2.fillPoly(mask, [face_points], 255)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if self.dilated:
            dilation_kernel = np.ones((35, 35), np.uint8)
            mask = cv2.dilate(mask, dilation_kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask


class HRNetLandmarkDetector:
    def __init__(self, device='cuda'):
        self.device = device
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=self.device)
        self.landmark_detector = facer.face_aligner('farl/wflw/448', device=self.device)

    def __call__(self, image):
        img_tensor = TF.to_tensor(image).to(self.device).unsqueeze(0) * 255.0
        
        with torch.inference_mode():
            faces = self.face_detector(img_tensor)
            if 'image_ids' not in faces or len(faces['image_ids']) == 0:
                return None
            faces = self.landmark_detector(img_tensor, faces)
            
        return faces['alignment'][0].cpu().numpy()


def draw_landmarks(image_size, landmarks):
    H, W = image_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    if landmarks is None:
        return Image.fromarray(canvas)

    pts = landmarks.astype(np.int32)

    def draw_points(indices, color):
        valid_indices = [i for i in indices if i < len(pts)]
        for i in valid_indices:
            x, y = pts[i]
            cv2.circle(canvas, (x, y), 3, color, -1)

    draw_points(range(0, 33), (255, 255, 255))
    draw_points(range(33, 42), (255, 255, 0))
    draw_points(range(42, 51), (255, 255, 0))
    draw_points(range(51, 55), (255, 0, 255))
    draw_points(range(55, 60), (255, 0, 255))
    
    if 76 < len(pts):
        draw_points(range(60, 68), (0, 255, 0))
        draw_points(range(68, 76), (0, 255, 0))
    
    if 97 < len(pts):
        draw_points(range(76, 88), (0, 0, 255))
        draw_points(range(88, 98), (0, 0, 255))

    return Image.fromarray(canvas)


def load_iresnet_model(model_path="glint360k_r100.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = iresnet100(pretrained=False, fp16=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def extract_embedding(model, device, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(img_tensor)
    
    return F.normalize(emb, p=2.0, dim=1)


def extract_embedding_insightface(app, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    faces = app.get(image)
    
    if len(faces) == 0:
        app.det_model.input_size = (512, 512)
        faces = app.get(image)
        
        if len(faces) == 0:
            app.det_model.input_size = (320, 320)
            faces = app.get(image)
        
        app.det_model.input_size = (640, 640)
        
        if len(faces) == 0:
            raise ValueError("No faces detected")
    
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    
    return faceid_embeds


def generate_caption(pil_image):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert face captioner. "
                        "Describe the face clearly and objectively, "
                        "including approximate age, perceived gender presentation, "
                        "hair style and color, expression, and any notable attributes. "
                        "Start the caption with 'A close-up photo of...' "
                        "Use one sentence and avoid names and guessing identity."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Describe this face in one sentence for a training caption."
                    ),
                },
            ],
        },
    ]
    
    text = qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = qwen_processor(
        text=text,
        images=[pil_image],
        return_tensors="pt",
    ).to(qwen_model.device)
    
    with torch.inference_mode():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    caption = qwen_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    
    return caption.strip()


def process_image(image_path, output_dir=None, model_path="glint360k_r100.pth", use_insightface=False, dilated=False):
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise ValueError(f"Image not found: {image_path}")
    
    if output_dir is None:
        output_dir = image_path.parent / f"conditions_{image_path.stem}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    pil_image = Image.open(image_path).convert("RGB")
    
    mask_extractor = FacialMaskExtractor(dilated=dilated)
    mask = mask_extractor.extract_mask(image)
    
    if mask is None:
        raise ValueError("No face detected in mask extraction")
    
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    
    if use_insightface:
        print(f"Loading InsightFace AntelopeV2...")
        app = FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        embedding = extract_embedding_insightface(app, image_path)
    else:
        print(f"Loading iResNet100 model...")
        iresnet_model, device = load_iresnet_model(model_path)
        embedding = extract_embedding(iresnet_model, device, image_path)
    
    embed_path = output_dir / f"{image_path.stem}.pt"
    torch.save(embedding, embed_path)
    
    print(f"Loading HRNet Landmark Detector...")
    landmark_detector = HRNetLandmarkDetector()
    landmarks = landmark_detector(pil_image)
    
    if landmarks is None:
        raise ValueError("No face detected in landmark extraction")
    
    landmark_image = draw_landmarks((pil_image.height, pil_image.width), landmarks)
    landmark_path = output_dir / f"{image_path.stem}_landmark.png"
    landmark_image.save(landmark_path)
    
    print(f"Generating caption...")
    caption = generate_caption(pil_image)
    caption_data = {"caption": caption}
    caption_path = output_dir / f"{image_path.stem}_caption.json"
    with open(caption_path, 'w', encoding='utf-8') as f:
        json.dump(caption_data, f, ensure_ascii=False, indent=2)
    
    print(f"Mask saved to: {mask_path}")
    print(f"Embedding saved to: {embed_path}")
    print(f"Landmark saved to: {landmark_path}")
    print(f"Caption saved to: {caption_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="checkpoints/glint360k_r100.pth")
    parser.add_argument("--insightface", action="store_true")
    parser.add_argument("--dilated", action="store_true", help="Use expanded/dilated facial mask with wider coverage")
    
    args = parser.parse_args()
    
    try:
        process_image(args.image_path, args.output_dir, args.model_path, args.insightface, args.dilated)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

