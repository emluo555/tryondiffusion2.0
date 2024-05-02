import torch
from torch.utils.data import DataLoader, Dataset

from tryondiffusionlite import TryOnImagenlite, get_unet_by_name_lite

from tryondiffusion import TryOnImagenTrainer

import os
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import save_image
import json

from dataset import DressCodeDataset

TRAIN_UNET_NUMBER = 1
BASE_UNET_IMAGE_SIZE = (128, 128) 
# SR_UNET_IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32
NUM_ITERATIONS = 1600000
TIMESTEPS = (256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    train_path = "/scratch/network/dg9272/dataset/lowerupper/train"
    valid_path = "/scratch/network/dg9272/dataset/lowerupper/test"
    train_dataset = DressCodeDataset(image_size=BASE_UNET_IMAGE_SIZE, path=train_path )
    valid_dataset = DressCodeDataset(image_size=BASE_UNET_IMAGE_SIZE, path=valid_path )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tryondiffusion_collate_fn,
    )
    validation_dataloader = DataLoader(
        valid_dataset,
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

    # Instantiate the Imagen model
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
        lr=1e-4,
        logname="lowerupper.csv"
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
