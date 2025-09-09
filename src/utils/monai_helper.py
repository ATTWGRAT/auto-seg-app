from monai.transforms import (
    Compose, LoadImaged, EnsureTyped, EnsureChannelFirstd, Orientationd,
    Spacingd, NormalizeIntensityd, ScaleIntensityd, Activationsd, AsDiscreted, Invertd
)

from monai.inferers import SlidingWindowInferer

from monai.data import DataLoader, Dataset

from monai.networks.nets import SegResNet
from huggingface_hub import hf_hub_download

import torch

import os

import shutil

def prepare_preprocessing():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureTyped(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=[3.0, 3.0, 3.0], mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0)
    ])

def prepare_postprocessing(preprocessing): 
    return Compose([
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        Invertd(
            keys="pred",
            transform=preprocessing,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True
        ),
    ])

def prepare_inferer():
    return SlidingWindowInferer(
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
        padding_mode="replicate",
        mode="gaussian",
        device="cpu"
    )
    
def prepare_dataloader(path, preprocessing):
    dataset = Dataset(data=[{"image": path}], transform=preprocessing)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

def prepare_network(model_path):
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to("cpu")

    model.load_state_dict(torch.load(model_path))

    model.eval()

    return model

def download_model_if_empty(model_dir="./model"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.listdir(model_dir):
        if not "model.pt" in os.listdir(model_dir):
            raise ValueError(f"Model directory {model_dir} is not empty but does not contain model.pt")
    else:
        print("Downloading model...")

        model_path = hf_hub_download(repo_id="MONAI/wholeBody_ct_segmentation", filename="models/model_lowres.pt", cache_dir=model_dir)
        dirs = os.listdir(model_dir)
        shutil.move(os.path.realpath(model_path), os.path.join(model_dir, "model.pt"))

        for dir in dirs:
            shutil.rmtree(os.path.join(model_dir, dir))