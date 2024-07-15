import torch, argparse
from torch import nn
from data_setup import create_dataloaders
from engine import train
from model_builder import build_effnet_v2_s, build_effnetb1
from utils import save_model

parser = argparse.ArgumentParser()
parser.add_argument("-lr", "--learning_rate", action="store_true", help="number of epochs")
parser.add_argument("-bs", "--batch_size", action="store_true", help="number of epochs")
parser.add_argument("-num", "--number_of_epochs", action="store_true", help="number of epochs")
parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

parser.add_argument("lr", type=float, help="learning rate")
parser.add_argument("bs", type=int, help="batch size")
parser.add_argument("n", type=int,  help="number of epochs")
args = parser.parse_args()
if args.verbose:
    print("learning_rate:", args.lr, "bs:", args.bs, "n:", args.n)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("Running on MPS device: ", mps_device)
else:
    print("MPS device not found.")

# Paths for training and testing dataset
# Split into 75 : 25
train_dir = "train/"
test_dir = "test/"

# train_path_list = list(train_dir.glob("*/*.png"))
# test_path_list = list(test_dir.glob("*/*.png"))

# PARAMETERS
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = args.bs
LEARNING_RATE = args.lr

# initialize model & weights
effnet_v2_s, effnet_v2_s_weights = build_effnet_v2_s(mps_device)

# format data for training
train_dataloader, test_dataloader = create_dataloaders(train_dir,
                                                       test_dir,
                                                       transform=effnet_v2_s_weights.transforms(),
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=NUM_WORKERS)

# initialize loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(effnet_v2_s.parameters(), lr=LEARNING_RATE)

experiment_number = 0

# Comparison between effnetv2_s and effnetb1
models = ["effnet_v2_s", "effnetb1"]
num_epochs = [5, 10]

for epochs in num_epochs:
    for model_name in models:
        experiment_number += 1
        print(f"[INFO] Experiment number: {experiment_number}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Number of epochs: {epochs}")

        # Select model
        if model_name == "effnet_v2_s":
            model, weights = build_effnet_v2_s(mps_device)
        else:
            model, weights = build_effnetb1(mps_device)

        # Get DataLoaders
        train_dataloader, test_dataloader = create_dataloaders(train_dir,
                                                               test_dir,
                                                               transform=weights.transforms(),
                                                               batch_size=32,
                                                               num_workers=0)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        train(model=model.to(mps_device),
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              epochs=epochs,
              device=mps_device)

        save_filepath = f"{model_name}_{epochs}_epochs.pth"
        save_model(model=model,
                   target_dir="models",
                   model_name=save_filepath)
        print("-" * 50 + "\n")

