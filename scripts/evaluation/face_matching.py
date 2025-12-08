import cv2
import numpy as np
import os
import shutil
import time

# ==================================================
# FINAL RESULTS
# ==================================================
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison1.jpg
#            Match: 69307.png (Score: 0.9986)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/69307.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison2.png
#            Match: 69126.png (Score: 0.9849)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/69126.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison3.png
#            Match: 69051.png (Score: 0.9894)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/69051.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison4.png
#            Match: 69557.png (Score: 0.9588)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/69557.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison5.png
#            Match: 68307.png (Score: 0.9860)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/68307.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison6.png
#            Match: 68126.png (Score: 0.9846)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/68126.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison7.png
#            Match: 68051.png (Score: 0.9911)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/68051.png
# ------------------------------
# ✅ MATCH | Query: /users/erluo/scratch/faceswap-diffusion/assets/comparison8.png
#            Match: 68557.png (Score: 0.9985)
#            Path:  /oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7/68557.png


QUERY_IMAGES = ["/users/erluo/scratch/faceswap-diffusion/assets/comparison1.jpg",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison2.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison3.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison4.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison5.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison6.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison7.png",
                "/users/erluo/scratch/faceswap-diffusion/assets/comparison8.png"] 
DATASET_ROOT = "/oscar/scratch/erluo/faceswap-diffusion/ffhq-dataset512/Part7" 


# Patch size for template matching (64 is standard for eyes/nose features)
PATCH_SIZE = 64 
# ---------------------

def crop_center_square(img):
    """Crops the largest possible square from the center to prevent distortion."""
    h, w = img.shape[:2]
    if h == w: return img
    size = min(h, w)
    y, x = (h - size) // 2, (w - size) // 2
    return img[y:y+size, x:x+size]

def pre_process_queries(image_paths):
    """
    Loads all query images, squares them, and extracts the search template.
    Returns a list of dictionaries containing the needed data for each query.
    """
    queries = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Warning: Could not read {path}. Skipping.")
            continue
            
        # 1. Square the image
        squared = crop_center_square(img)
        q_size = squared.shape[0] # Height/Width (same now)
        
        # 2. Extract Template (Center Patch)
        c = q_size // 2
        d = PATCH_SIZE // 2
        template = squared[c-d:c+d, c-d:c+d]
        
        queries.append({
            "id": idx,
            "filename": path,
            "q_size": q_size,
            "template": template,
            "best_score": -1.0,
            "best_match_file": None,
            "best_match_path": None,
            "found": False
        })
    return queries

def find_matches_batch():
    # 1. Prepare all queries
    active_queries = pre_process_queries(QUERY_IMAGES)
    if not active_queries:
        print("No valid query images found.")
        return

    print(f"Loaded {len(active_queries)} query images.")
    print(f"Scanning {DATASET_ROOT} (Single Pass)...")
    
    start_time = time.time()
    count = 0
    
    # 2. Single Pass through Dataset
    for root, dirs, files in os.walk(DATASET_ROOT):
        # Optimization: If all queries are found, stop scanning immediately
        if all(q['found'] for q in active_queries):
            print("\nAll images found! Stopping search.")
            break
            
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            count += 1
            if count % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"Scanned {count} images in {elapsed:.1f}s...")

            db_path = os.path.join(root, filename)
            db_img = cv2.imread(db_path)
            if db_img is None: continue

            # 3. Check this dataset image against ALL active queries
            for q in active_queries:
                # If we already found a perfect match for this query, skip it
                if q['found']: continue

                # Resize dataset image to match this specific query's dimension
                # (Only resize if dimensions differ to save CPU)
                if db_img.shape[0] != q['q_size']:
                    db_img_resized = cv2.resize(db_img, (q['q_size'], q['q_size']))
                else:
                    db_img_resized = db_img

                # Template Match
                res = cv2.matchTemplate(db_img_resized, q['template'], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                # Update best score for this query
                if max_val > q['best_score']:
                    q['best_score'] = max_val
                    q['best_match_file'] = filename
                    q['best_match_path'] = db_path
                    
                    # Check for "Perfect Match" (Early Exit for this specific face)
                    if max_val > 0.995:
                        q['found'] = True
                        print(f"   ✓ Found match for {q['filename']}: {filename} (Score: {max_val:.4f})")

    # 4. Format Output
    print("\n" + "="*50)
    print(f"FINAL RESULTS")
    print("="*50)
    
    results_array = []
    
    for q in active_queries:
        result_entry = {
            "query_image": q['filename'],
            "found_match": q['best_match_file'],
            "match_path": q['best_match_path'],
            "correlation": q['best_score']
        }
        results_array.append(result_entry)
        
        # Print readable status
        status = "✅ MATCH" if q['best_score'] > 0.9 else "⚠️ LOW CONFIDENCE"
        print(f"{status} | Query: {q['filename']}")
        print(f"           Match: {q['best_match_file']} (Score: {q['best_score']:.4f})")
        print(f"           Path:  {q['best_match_path']}")
        print("-" * 30)

        # Optional: Save copy
        # if q['best_match_path']:
        #     save_name = f"FOUND_for_{q['filename']}.png"
        #     shutil.copy(q['best_match_path'], save_name)

    return results_array

if __name__ == "__main__":
    final_results = find_matches_batch()
    # final_results is your array of objects/dictionaries