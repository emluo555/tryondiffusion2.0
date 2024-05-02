import torch
from torch.utils.data import DataLoader, Dataset

from tryondiffusionlite import TryOnImagenlite, get_unet_by_name_lite

from tryondiffusion import TryOnImagenTrainer

import os
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image
import json

TRAIN_UNET_NUMBER = 1
BASE_UNET_IMAGE_SIZE = (128, 128) 
# SR_UNET_IMAGE_SIZE = (64, 64)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_ITERATIONS = 50000
TIMESTEPS = (256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SyntheticTryonDataset(Dataset):
    def __init__(self, image_size, pose_size=(18, 2)):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            image_size (tuple): The height and width of the images (height, width).
            pose_size (tuple): The size of the pose tensors (default: (18, 2)).
        """
        self.data = []
        for filename in os.listdir('/scratch/network/dg9272/cos485/lowerupper/image'):
            f = os.path.join('/scratch/network/dg9272/cos485/lowerupper/image', filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.data.append(filename[:-6])
        self.num_samples = len(self.data)
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        person_image = read_image(f"/scratch/network/dg9272/cos485/lowerupper/image/{item}_0.jpg").float() / 255
        person_image = v2.Resize(size=self.image_size)(person_image)
        
        ca_image = read_image(f"/scratch/network/dg9272/cos485/lowerupper/agnostic/{item}_0.jpg").float() / 255
        ca_image = v2.Resize(size=self.image_size)(ca_image)
        
        garment_image = read_image(f"/scratch/network/dg9272/cos485/lowerupper/cloth/{item}_1.jpg").float() / 255
        garment_image = v2.Resize(size=self.image_size)(garment_image)
        
        f = open(f"/scratch/network/dg9272/cos485/lowerupper/openpose_json/{item}_2.json")
        points = json.load(f)['keypoints']
        person_pose = torch.tensor([[ points[i * 2 + 1] * 2 / 1024, points[i * 2] * 2 / 768] for i in range(18)])
        f.close()
        
        # garment_pose = torch.randn(*(1, 2))

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


def main():
    print("Instantiating the dataset and dataloader...")
    dataset = SyntheticTryonDataset(image_size=BASE_UNET_IMAGE_SIZE if TRAIN_UNET_NUMBER == 2 else BASE_UNET_IMAGE_SIZE
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tryondiffusion_collate_fn,
    )
    validation_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tryondiffusion_collate_fn,
    )
    print("Checking the dataset and dataloader...")
    sample = next(iter(train_dataloader))
    for k, v in sample.items():
        print(f"{k}: {v.shape}")
    # Instantiate the unets
    print("Instantiating U-Nets...")
    base_unet = get_unet_by_name_lite("base")
    # # sr_unet = get_unet_by_name("sr")

    # # Instantiate the Imagen model
    imagen = TryOnImagenlite(
        unets=(base_unet),
        image_sizes=(BASE_UNET_IMAGE_SIZE, ),
        timesteps=TIMESTEPS,
    )
    print("Instantiating the trainer...")
    trainer = TryOnImagenTrainer(
        imagen=imagen,
        max_grad_norm=1.0,
        accelerate_cpu=False,
        accelerate_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        device="cuda",
        checkpoint_path="/scratch/network/dg9272/cos485/checkpoints/lowerupper",
        checkpoint_every=100,
        lr=1e-6
    )

    print("Starting sampling loop...")
    trainer.add_train_dataloader(train_dataloader)
    trainer.add_valid_dataloader(validation_dataloader)
    validation_sample = next(iter(validation_dataloader))
    save_image(validation_sample['person_images'], 'out/person.jpg')
    save_image(validation_sample['ca_images'], 'out/ca.jpg')
    save_image(validation_sample['garment_images'], 'out/garment.jpg')
    del validation_sample['person_images']
    imagen_sample_kwargs = dict(
        **validation_sample,
        batch_size=BATCH_SIZE,
        cond_scale=2.0,
        start_at_unet_number=1,
        return_all_unet_outputs=True,
        return_pil_images=True,
        use_tqdm=True,
        use_one_unet_in_gpu=True,
    )

    images = trainer.sample(**imagen_sample_kwargs)  # returns List[Image]
    # assert len(images) == 2
    # assert len(images[0]) == BATCH_SIZE and len(images[1]) == BATCH_SIZE

    for unet_output in images:
        for image in unet_output:
            image.save("out/test.jpg") 


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen_trainer.py
    main()
