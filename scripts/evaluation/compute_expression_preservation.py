# Compute expression preservation using InsightFace landmarks
# follows methodology from DreamID paper of L2 distance between the target and swapped

import argparse
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

EXPRESSION_INDICES = [0, 1, 3, 4]  # focus on eyes and mouth corners

def compute_expression_l2_distance(lm_target, lm_swapped, expression_indices=None):
    """Compute L2 distance between landmarks (following DreamID paper) (lower means better preservation)"""
    if expression_indices is not None:
        lm_target = lm_target[expression_indices]
        lm_swapped = lm_swapped[expression_indices]

    # l2
    distance = np.linalg.norm(lm_target - lm_swapped)
    return distance

def normalize_landmarks(landmarks, bbox):
    """Normalize landmarks by face bounding box size
    Makes metric scale-invariant
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    normalized = landmarks.copy()
    normalized[:, 0] = (normalized[:, 0] - x1) / width
    normalized[:, 1] = (normalized[:, 1] - y1) / height

    return normalized

def main():
    parser = argparse.ArgumentParser(
        description="Compute expression preservation using InsightFace landmarks"
    )
    parser.add_argument("pairs_json", type=str,
                       help="JSON file with target -> swapped mappings")
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--swapped_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="expression_preservation.json")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize by face bbox (default: True)")

    args = parser.parse_args()

    print("Loading InsightFace model...")
    app = FaceAnalysis(
        name='antelopev2',
        providers=['CUDAExecutionProvider'] if args.gpu >= 0 else ['CPUExecutionProvider']
    )
    app.prepare(ctx_id=args.gpu if args.gpu >= 0 else -1, det_size=(640, 640))
    print("InsightFace loaded")

    print(f"\nLoading pairs from {args.pairs_json}...")
    with open(args.pairs_json) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    results = {}
    distances = []
    failed = 0

    for target_name, swapped_name in tqdm(pairs.items(), desc="Computing expression L2"):
        target_path = Path(args.target_dir) / target_name
        swapped_path = Path(args.swapped_dir) / swapped_name

        if not target_path.exists() or not swapped_path.exists():
            results[target_name] = {'l2_distance': None, 'error': 'file couldnt be found'}
            failed += 1
            continue

        # load images
        target_img = cv2.imread(str(target_path))
        swapped_img = cv2.imread(str(swapped_path))

        if target_img is None or swapped_img is None:
            results[target_name] = {'l2_distance': None, 'error': 'loading failure'}
            failed += 1
            continue

        try:
            # get landmarks
            faces_target = app.get(target_img)
            faces_swapped = app.get(swapped_img)

            if len(faces_target) == 0 or len(faces_swapped) == 0:
                results[target_name] = {'l2_distance': None, 'error': 'No face detected'}
                failed += 1
                continue

            face_target = faces_target[0]
            face_swapped = faces_swapped[0]

            lm_target = face_target.kps  # Shape (5, 2)
            lm_swapped = face_swapped.kps

            # normalize by face bbox if requested
            if args.normalize:
                lm_target = normalize_landmarks(lm_target, face_target.bbox)
                lm_swapped = normalize_landmarks(lm_swapped, face_swapped.bbox)

            # Compute L2 distance on expression landmarks (eyes + mouth)
            l2_dist = compute_expression_l2_distance(
                lm_target,
                lm_swapped,
                expression_indices=EXPRESSION_INDICES
            )

            distances.append(l2_dist)
            results[target_name] = {
                'l2_distance': float(l2_dist),
                'num_landmarks_used': len(EXPRESSION_INDICES)
            }

        except Exception as e:
            results[target_name] = {'l2_distance': None, 'error': str(e)}
            failed += 1

    # stats
    if distances:
        summary = {
            'mean_l2_distance': float(np.mean(distances)),
            'std_l2_distance': float(np.std(distances)),
            'median_l2_distance': float(np.median(distances)),
            'min_l2_distance': float(np.min(distances)),
            'max_l2_distance': float(np.max(distances)),
            'total_pairs': len(distances),
            'failed_pairs': failed,
            'normalized': args.normalize,
            'method': 'InsightFace landmarks (eyes + mouth)',
            'note': 'Lower L2 = better expression preservation (DreamID methodology)'
        }
    else:
        summary = {'error': 'No valid measurements', 'failed_pairs': failed}

    output_data = {
        'summary': summary,
        'per_image': results
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("Expression preservation results")

    if 'mean_l2_distance' in summary:
        print(f"Mean L2 Distance:     {summary['mean_l2_distance']:.4f}")
        print(f"Std L2 Distance:      {summary['std_l2_distance']:.4f}")
        print(f"Median L2 Distance:   {summary['median_l2_distance']:.4f}")
        print(f"Total pairs:          {summary['total_pairs']}")
        print(f"Failed pairs:         {summary['failed_pairs']}")
    else:
        print(f"No valid measurements (failed: {failed})")

    print(f"\n Results saved to: {output_path}")

if __name__ == "__main__":
    main()