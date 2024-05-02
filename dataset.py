import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.io import read_image
from torchvision.transforms import v2
import json

class DressCodeDataset(Dataset):
    def __init__(self, image_size, path):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (tuple): The height and width of the images (height, width).
            pose_size (tuple): The size of the pose tensors (default: (18, 2)).
        """
        self.data = []
        self.imgpath = os.path.join(path, "image")
        self.clothpath = os.path.join(path, "cloth")
        self.keypointpath = os.path.join(path, "keypoints")
        self.agnosticpath = os.path.join(path, "agnostic")
        for filename in os.listdir(self.imgpath):
            f = os.path.join(self.imgpath, filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.data.append(filename[:-6])
        self.num_samples = len(self.data)
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        person_image = read_image(os.path.join(self.imgpath, f"{item}_0.jpg")).float() / 255
        person_image = v2.Resize(size=self.image_size)(person_image)
        
        ca_image = read_image(os.path.join(self.agnosticpath, f"{item}_0.jpg")).float() / 255
        ca_image = v2.Resize(size=self.image_size)(ca_image)
        garment_image = read_image(os.path.join(self.clothpath, f"{item}_1.jpg")).float() / 255
        garment_image = v2.Resize(size=self.image_size)(garment_image)
        
        f = open(os.path.join(self.keypointpath, f"{item}_2.json"))
        points = json.load(f)['keypoints']
        person_pose = torch.tensor([[ points[i][1] / 1024, points[i][0] / 768] for i in range(18)])
        f.close()
        
        sample = {
            "person_images": person_image,
            "ca_images": ca_image,
            "garment_images": garment_image,
            "person_poses": person_pose,
            # "garment_poses": garment_pose,
        }

        return sample


def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        # "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }   
if __name__ == "__main__":
    train_path = "/scratch/network/dg9272/cos485/dataset/upper/train"
    valid_path = "/scratch/network/dg9272/cos485/dataset/upper/test"
    train_dataset = DressCodeDataset(image_size=(128,128), path=train_path )
    valid_dataset = DressCodeDataset(image_size=(128,128), path=valid_path )
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=tryondiffusion_collate_fn,
        )
    validation_dataloader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=tryondiffusion_collate_fn,
    )
    next(iter(train_dataloader))
    next(iter(validation_dataloader))