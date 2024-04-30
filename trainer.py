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
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
NUM_ITERATIONS = 100000
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
        for filename in os.listdir('/scratch/network/dg9272/cos485/VITON-HD/train/image/'):
            f = os.path.join('/scratch/network/dg9272/cos485/VITON-HD/train/image/', filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.data.append(filename[:-4])
        self.num_samples = len(self.data)
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        person_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/image/{item}.jpg").float() / 225
        person_image = v2.Resize(size=self.image_size)(person_image)
        
        ca_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/agnostic-v3.2/{item}.jpg").float() / 225
        ca_image = v2.Resize(size=self.image_size)(ca_image)
        
        garment_image = read_image(f"/scratch/network/dg9272/cos485/VITON-HD/train/cloth/{item}.jpg").float() /225
        garment_image = v2.Resize(size=self.image_size)(garment_image)
        
        f = open(f"/scratch/network/dg9272/cos485/VITON-HD/train/openpose_json/{item}_keypoints.json")
        data = json.load(f)['people'][0]['pose_keypoints_2d']
        person_pose = torch.tensor([[ data[i * 3 + 1] / 1024, data[i * 3] / 768] for i in range(25)])
        # print(person_pose)
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
        shuffle=False,
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
        checkpoint_path="/scratch/network/dg9272/cos485/checkpoints/checks5",
        checkpoint_every=100,
        lr=1e-6
    )

    trainer.add_train_dataloader(train_dataloader)
    trainer.add_valid_dataloader(validation_dataloader)
    print("Starting training loop...")
    # training loop
    for i in range(NUM_ITERATIONS):
        # TRAINING
        trainer.train_step(unet_number=TRAIN_UNET_NUMBER)
        # if (i % NUM_ITERATIONS == 0):
        #     trainer.valid_step(unet_number=TRAIN_UNET_NUMBER)

    # SAMPLING
    print("Starting sampling loop...")
    validation_sample = next(trainer.valid_dl_iter)
    _ = validation_sample.pop("person_images")
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
            image.save("output.png") 


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen_trainer.py
    main()
