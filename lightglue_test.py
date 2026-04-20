from lightglue import LightGlue, SuperPoint
# superpoint is feature extracter and lightglue is the matcher
from lightglue.utils import load_image, rbd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device) # feature extractor
matcher = LightGlue(features='superpoint').eval().to(device) 

image0 = load_image('assets/DSC_0410.jpg').to(device)
image1 = load_image('assets/DSC_0411.jpg').to(device)

feats0 = extractor.extract(image0) # run superpoint on each image
feats1 = extractor.extract(image1)

matches01 = matcher({'image0': feats0, 'image1': feats1}) # run lightglue on the feature sets. contains matcher output
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] # remove the batch dimension (changing formatting)

matches = matches01['matches'] # get actual match list
points0 = feats0['keypoints'][matches[..., 0]]
points1 = feats1['keypoints'][matches[..., 1]]

print("Number of matches:", len(matches)) # number of point correspondences
print("Stop layer:", matches01.get("stop", None))