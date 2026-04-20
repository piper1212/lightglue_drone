from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
## for transformation matrix:
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

# Drone 1 image (this is the one we rotate)
image0 = load_image('assets/DSC_0410.jpg').to(device)

# Drone 2 image (this stays fixed)
image1 = load_image('assets/rotated.jpg').to(device) # DSC_0411

angles = [0, 90, 180, 270]
results = []

for angle in angles:
    # rotate image0 by multiples of 90 degrees
    k = angle // 90
    rotated_image0 = torch.rot90(image0, k=k, dims=(1, 2))

    # run SuperPoint on rotated image0 and fixed image1
    feats0 = extractor.extract(rotated_image0)
    feats1 = extractor.extract(image1)

    # run LightGlue on the feature sets
    matches01 = matcher({'image0': feats0, 'image1': feats1})

    # remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # get actual match l
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]] # coords of matched keypoints in first image
    points1 = feats1['keypoints'][matches[..., 1]] # coords of corresponding matched keypoints in second image
    # c2v.findHomography will need these coordinate pairs

    num_matches = len(matches)
    stop_layer = matches01.get("stop", None)

    results.append({
        "angle": angle,
        "num_matches": num_matches,
        "stop_layer": stop_layer,
        "points0": points0,
        "points1": points1,
        "matches": matches
    })

    print(f"Angle: {angle} degrees")
    print(f"  Number of matches: {num_matches}")
    print(f"  Stop layer: {stop_layer}")

# pick the angle with the highest number of matches
best_result = max(results, key=lambda x: x["num_matches"])

print("\nBest rotation angle:", best_result["angle"])
print("Best number of matches:", best_result["num_matches"])
print("Best stop layer:", best_result["stop_layer"])

# compute transformation matrix for the best angle
# to get geometric mapping for drone 1 image to drone 2 image (use RANSAC)
pts0 = best_result["points0"].cpu().numpy()
pts1 = best_result["points1"].cpu().numpy()

if len(pts0) < 4:
    print("Not enough matches to compute transformation matrix.")
else:
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

    if H is None:
        print("Transformation matrix computation failed.")
    else:
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = num_inliers / len(pts0)

        print("\nTransformation matrix (H):")
        print(H)
        print("Total matches used:", len(pts0))
        print("RANSAC inliers:", num_inliers)
        print("Inlier ratio:", inlier_ratio)

# common view area model - use transformation matrix to find the
