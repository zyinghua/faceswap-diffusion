#Unified eval script that runs expression, id similarity, fid, and id retrieval

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from compute_expression_preservation import compute_expression_l2_distance, normalize_landmarks
    from compute_id_similarity import IDSimilarityCalculator
    from compute_fid import compute_metrics
    from compute_id_retrieval import calculate_retrieval_accuracy
except ImportError:
    print("ERROR: Could not import evaluation scripts")
    sys.exit(1)

import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def run_expression_preservation(args, app):
    """Run expression preservation metric using landmarks"""
    print("Metric 1: Computing Expression Preservation (InsightFace landmarks)")
    
    with open(args.pairs_json) as f:
        pairs = json.load(f)
    
    distances = []
    failed = 0
    
    EXPRESSION_INDICES = [0, 1, 3, 4]  # eyes and mouth
    
    for target_name, swapped_name in tqdm(pairs.items(), desc="Expression L2"):
        target_path = Path(args.target_dir) / target_name
        swapped_path = Path(args.swapped_dir) / swapped_name
        
        if not target_path.exists() or not swapped_path.exists():
            failed += 1
            continue
        
        target_img = cv2.imread(str(target_path))
        swapped_img = cv2.imread(str(swapped_path))
        
        if target_img is None or swapped_img is None:
            failed += 1
            continue
        
        try:
            faces_target = app.get(target_img)
            faces_swapped = app.get(swapped_img)
            
            if len(faces_target) == 0 or len(faces_swapped) == 0:
                failed += 1
                continue
            
            lm_target = faces_target[0].kps
            lm_swapped = faces_swapped[0].kps
            
            lm_target = normalize_landmarks(lm_target, faces_target[0].bbox)
            lm_swapped = normalize_landmarks(lm_swapped, faces_swapped[0].bbox)

            l2_dist = compute_expression_l2_distance(
                lm_target, lm_swapped, EXPRESSION_INDICES
            )
            distances.append(l2_dist)
            
        except Exception:
            failed += 1
    
    if distances:
        result = {
            'mean_l2_distance': float(np.mean(distances)),
            'std_l2_distance': float(np.std(distances)),
            'median_l2_distance': float(np.median(distances)),
            'min_l2_distance': float(np.min(distances)),
            'max_l2_distance': float(np.max(distances)),
            'total_pairs': len(distances),
            'failed_pairs': failed
        }
        print(f"Mean L2 Distance: {result['mean_l2_distance']:.4f}")
        print(f"Median L2: {result['median_l2_distance']:.4f}")
        print(f"Success: {len(distances)}/{len(pairs)}")
    else:
        result = {'error': 'No valid measurements', 'failed_pairs': failed}
        print(f"All pairs failed")
    
    return result


def run_id_similarity(args):
    """Run ID similarity metric using ArcFace"""
    print("Metric2: Computing ID Similarity (ArcFace)")
    
    try:
        calculator = IDSimilarityCalculator(use_gpu=(args.gpu >= 0))
        id_results, _ = calculator.compute_similarities_from_json(
            args.pairs_json,
            args.source_dir,
            args.swapped_dir
        )

        similarities = [sim for sim, _, _ in id_results.values() if sim is not None]
        
        if similarities:
            result = {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'median': float(np.median(similarities)),
                'preservation@0.3': float(sum(s >= 0.3 for s in similarities) / len(similarities)),
                'preservation@0.5': float(sum(s >= 0.5 for s in similarities) / len(similarities)),
                'preservation@0.7': float(sum(s >= 0.7 for s in similarities) / len(similarities)),
                'total_pairs': len(similarities),
                'failed_pairs': len(id_results) - len(similarities)
            }
            print(f"Mean ID Similarity: {result['mean']:.4f}")
            print(f"Preservation@0.5: {result['preservation@0.5']*100:.2f}%")
            print(f"Preservation@0.7: {result['preservation@0.7']*100:.2f}%")
        else:
            result = {'error': 'No valid similarities'}
            print(f"No valid similarities computed")
            
    except Exception as e:
        result = {'error': str(e)}
        print(f"Failed: {e}")
    
    return result


def run_id_retrieval(args):
    """Run ID retrieval metric (Top-1 and Top-5 accuracy)"""
    print("Metric 3: Computing ID Retrieval Accuracy")
    
    if args.retrieval_db_json is None:
        print("Skipping: --retrieval_db_json not provided")
        return {'skipped': True, 'reason': 'retrieval_db_json not provided'}
    
    if not Path(args.retrieval_db_json).exists():
        print(f"Skipping: {args.retrieval_db_json} not found")
        return {'skipped': True, 'reason': 'retrieval_db_json file not found'}
    
    try:
        top1_acc, top5_acc = calculate_retrieval_accuracy(
            args.pairs_json,
            args.retrieval_db_json,
            ctx_id=args.gpu
        )
        
        result = {
            'top1_accuracy': float(top1_acc),
            'top5_accuracy': float(top5_acc),
            'top1_percentage': float(top1_acc * 100),
            'top5_percentage': float(top5_acc * 100)
        }
        print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")
        
    except Exception as e:
        result = {'error': str(e)}
        print(f"Failed: {e}")
    
    return result


def run_fid(args):
    """Run FID, CLIP-FID, and KID metrics"""
    print("Metric 4: Computing Image Quality Metrics (FID, CLIP-FID, KID)")
    
    try:
        quality_metrics = compute_metrics(
            args.target_dir,
            args.swapped_dir,
            batch_size=args.batch_size,
            device='cuda' if args.gpu >= 0 else 'cpu'
        )
        
        result = {
            'fid': float(quality_metrics['fid']),
            'clip_fid': float(quality_metrics['clip_fid']),
            'kid': float(quality_metrics['kid'])
        }
        print(f"FID: {quality_metrics['fid']:.4f}")
        print(f"CLIP-FID: {quality_metrics['clip_fid']:.4f}")
        print(f"KID: {quality_metrics['kid']:.6f}")
        
    except Exception as e:
        result = {'error': str(e)}
        print(f"Failed: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation suite for face swap model"
    )
    
    # required
    parser.add_argument("--source_dir", type=str, required=True,
                       help="Directory with source identity images")
    parser.add_argument("--target_dir", type=str, required=True,
                       help="Directory with target images (ground truth)")
    parser.add_argument("--swapped_dir", type=str, required=True,
                       help="Directory with generated face swaps")
    
    
    parser.add_argument("--pairs_json", type=str, required=True,
                       help="JSON mapping source/target -> swapped image")
    
    # optional
    parser.add_argument("--retrieval_db_json", type=str, default=None,
                       help="JSON mapping embedding_id -> source image (for retrieval metric)")
    
    # Output
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output JSON file for all results")
    
    # optional args
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device ID (-1 for CPU)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for FID computation")
    
    args = parser.parse_args()
    
    for path_name, path_val in [
        ("source_dir", args.source_dir),
        ("target_dir", args.target_dir),
        ("swapped_dir", args.swapped_dir),
        ("pairs_json", args.pairs_json)
    ]:
        if not Path(path_val).exists():
            print(f"ERROR: {path_name} does not exist: {path_val}")
            sys.exit(1)
    
    print("Running end-to-end eval metric script")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'source_dir': args.source_dir,
            'target_dir': args.target_dir,
            'swapped_dir': args.swapped_dir,
            'pairs_json': args.pairs_json,
            'retrieval_db_json': args.retrieval_db_json,
            'gpu': args.gpu
        }
    }
    
    app = FaceAnalysis(
        name='antelopev2',
        providers=['CUDAExecutionProvider'] if args.gpu >= 0 else ['CPUExecutionProvider']
    )
    app.prepare(ctx_id=args.gpu if args.gpu >= 0 else -1, det_size=(640, 640))
    
    # run metrics
    results['expression_preservation'] = run_expression_preservation(args, app)
    results['id_similarity'] = run_id_similarity(args)
    results['id_retrieval'] = run_id_retrieval(args)
    results['quality_metrics'] = run_fid(args)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    

    print("Finished eval script")
    print(f"\nResults saved to: {output_path}")
    
    # Expression
    if 'mean_l2_distance' in results['expression_preservation']:
        exp = results['expression_preservation']
        print(f"Expression Preservation (L2):  {exp['mean_l2_distance']:.4f}")
    
    # ID Similarity
    if 'mean' in results['id_similarity']:
        id_sim = results['id_similarity']
        print(f"ID Similarity (mean):          {id_sim['mean']:.4f}")
        print(f"  └─ Preservation@0.5:         {id_sim['preservation@0.5']*100:.1f}%")
        print(f"  └─ Preservation@0.7:         {id_sim['preservation@0.7']*100:.1f}%")
    
    # ID Retrieval
    if 'top1_accuracy' in results['id_retrieval']:
        retr = results['id_retrieval']
        print(f"ID Retrieval Top-1:            {retr['top1_percentage']:.1f}%")
        print(f"ID Retrieval Top-5:            {retr['top5_percentage']:.1f}%")
    elif results['id_retrieval'].get('skipped'):
        print(f"ID Retrieval:                  Skipped (no retrieval DB)")
    
    # Quality
    if 'fid' in results['quality_metrics']:
        qual = results['quality_metrics']
        print(f"FID Score:                     {qual['fid']:.2f}")
        print(f"CLIP-FID:                      {qual['clip_fid']:.2f}")
        print(f"KID:                           {qual['kid']:.6f}")
    
    print("\n All metrics finished")


if __name__ == "__main__":
    main()