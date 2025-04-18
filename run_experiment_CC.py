#%%
import time
import os
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from tifffile import imwrite
from dataset_CC import RGB2ThermalDataset   # reads CSV → 1‑ch PIL
from model import Pix2Pix              
import numpy as np

# Save folder for generated images from validation loop
SAVE_DIR = "predictionsPIX2PIX_CC_final"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "model_current_CC_final"
RUN_NAME   = "pix2pix-CC_final"
#%%
def train(img_dataloader, model):
    loop = tqdm(img_dataloader, leave=True)
    total_G = 0.0
    total_D = 0.0
    n_batches = 0

    for idx, (x, y, imgidx) in enumerate(loop):
        model.set_input(x, y)
        model.optimize_parameters()

        # grab the raw losses
        losses = model.get_current_losses()
        wandb.log({
            'G_GAN':  losses['G_GAN'],
            'G_L1':   losses['G_L1'],
            'D_real': losses['D_real'],
            'D_fake': losses['D_fake']
        })

        # accumulate for averaging
        total_G += model.loss_G.item()
        total_D += model.loss_D.item()
        n_batches += 1
        loop.set_postfix(batch_G_loss=model.loss_G.item())

    # compute epoch means
    avg_G = total_G / n_batches
    avg_D = total_D / n_batches
    return avg_G, avg_D

def validate(img_dataloader, model):
    model.netG.to(model.device)
    loop = tqdm(img_dataloader, leave=True)

    for idx, (x, y, imgidx) in enumerate(loop):
        # Set inputs and run forward pass
        model.set_input(x, y)
        model.test()

        # Get the single‑channel prediction as a NumPy array
        pred = model.fake.squeeze(0).squeeze(0).cpu().numpy()
        pred = (pred * 255).astype("uint8")
        img = np.array(pred)

        # Unwrap the Subset to access the original dataset's get_filenames()
        base_ds = img_dataloader.dataset.dataset
        csv_path = base_ds.get_filenames(imgidx)
        fn = os.path.splitext(os.path.basename(csv_path))[0]

        # Save and log the prediction
        out_path = os.path.join(SAVE_DIR, f"{fn}_fake.tiff")
        imwrite(out_path, img.astype('uint8'))
        wandb.log({
            "Validation Prediction": wandb.Image(out_path, caption=fn)
        })
#%%
if __name__ == '__main__':
  #%%
    # hyperparameters 
    in_chan       = 3     # RGB
    out_chan      = 1     # thermal
    learning_rate = 1e-4
    batch_size    = 6
    num_epochs    = 100

    # init W&B
    run = wandb.init(
        project=RUN_NAME,
        job_type="train",
        config={
            "learning_rate": learning_rate,
            "batch_size":    batch_size,
            "num_epochs":    num_epochs
        },
        notes="Training Pix2Pix for RGB→thermal translation using CSV targets"
    )
#%%
    # build & setup model
    model = Pix2Pix(
        in_channels=in_chan,
        out_channels=out_chan,
        isTrain=True,
        gan_mode='lsgan',
        device='cuda',
        lr=learning_rate
    )
    model.setup()

    # simple resize transform to 240×240
    def transform(rgb_img, tir_img):
        size = (240, 240)
        return rgb_img.resize(size), tir_img.resize(size)

    # dataset & loaders 
    # file path to folder containing both .png and .csv files
    folder = r"C:\Users\Const\OneDrive\Desktop\PhD\Comp_Photo\Renamed_Files_paired"
    ds = RGB2ThermalDataset(
        rgb_pattern     = os.path.join(folder, "*.png"),
        thermal_pattern = os.path.join(folder, "*.csv"),
        transform       = transform
    )

    # debug: ensure files were found
    print(f"Found {len(ds)} samples "
          f"(RGB: {len(ds.rgb_paths)}, CSV: {len(ds.thermal_paths)})")

    # split into training set & validation set (70/30)
    train_size = int(len(ds) * 0.7)
    val_size   = len(ds) - train_size
    training_set, validation_set = torch.utils.data.random_split(
        ds, [train_size, val_size]
    )

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(validation_set,   batch_size=1,         shuffle=False)
#%%
    # save loaders for reproducibility
    #pkl.dump(train_loader, open("train_loader.pkl", "wb"))
    #pkl.dump(val_loader,   open("val_loader.pkl",   "wb"))
    #run.save("train_loader.pkl")
    #run.save("val_loader.pkl")

    # training loop 
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        model.netG.to(model.device)
        model.netD.to(model.device)

        avg_G, avg_D = train(train_loader, model)

        validate(val_loader, model)

        # print loss averages
        print(f" → Epoch {epoch} avg G loss: {avg_G:.4f}, avg D loss: {avg_D:.4f}")
        # (optional) also log these to W&B
        wandb.log({'epoch_avg_G': avg_G, 'epoch_avg_D': avg_D, 'epoch': epoch})

        model.save_networks(epoch)
        model.update_learning_rate()

    # log final models to W&B
    run.log_model('model_current_D.pth', name=RUN_NAME + '-D')
    run.log_model('model_current_G.pth', name=RUN_NAME + '-G')

    print("Training complete\n")

    # final validation
    print("Running final validation...")
    validate(val_loader, model)
    print("Validation complete")


        