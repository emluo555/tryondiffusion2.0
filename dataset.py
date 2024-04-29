import os
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image
import torch
import json
image_size = (128, 128)
data = []
for filename in os.listdir('/scratch/network/dg9272/cos485/VITON-HD/train/paired-ck-point/'):
    f = os.path.join('/scratch/network/dg9272/cos485/VITON-HD/train/paired-ck-point/', filename)
    # checking if it is a file
    if os.path.isfile(f):
        data.append(filename[:-5])
for i in range(1):
    if os.path.isfile(f"/scratch/network/dg9272/cos485/VITON-HD/train/image/{data[i]}.jpg"):
        person_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/image/{data[i]}.jpg").float() / 225
        print(person_image.shape)
        person_image = v2.Resize(size=image_size)(person_image)
    if os.path.isfile(f"/scratch/network/dg9272/cos485/VITON-HD/train/cloth/{data[i]}.jpg"):
        garment_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/cloth/{data[i]}.jpg").float() / 225
        garment_image = v2.Resize(size=image_size)(garment_image)
    if os.path.isfile(f"/scratch/network/dg9272/cos485/VITON-HD/train/agnostic-v3.2/{data[i]}.jpg"):
        ca_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/agnostic-v3.2/{data[i]}.jpg").float() / 225
        ca_image = v2.Resize(size=image_size)(ca_image)
    if os.path.isfile(f"/scratch/network/dg9272/cos485/VITON-HD/train/openpose_json/{data[i]}_keypoints.json"):
        f = open(f"/scratch/network/dg9272/cos485/VITON-HD/train/openpose_json/{data[i]}_keypoints.json")
        data = json.load(f)['people'][0]['pose_keypoints_2d']
        person_pose = torch.tensor([[ data[i * 3 + 1] / 1024, data[i * 3] / 768] for i in range(25)])
        print(person_pose)
        f.close()
    if os.path.isfile(f"/scratch/network/dg9272/cos485/VITON-HD/train/paired-ck-point/{data[i]}.json"):
        print('cry')
        print(i)

# with open('/scratch/network/dg9272/cos485/VITON-HD/train_pairs.txt') as f:
#     s = f.read().splitlines()
    
#     for pair in s:
#         pairs.append(*pair.split(' '))
# print(len(pairs))
