import matplotlib.pyplot as plt
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import TensorDataset, DataLoader
import argparse

from scripts.networks import InvNet
from scripts.training import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="NeurIPS Experiments")
parser.add_argument("--layers", default=5,type=int)
parser.add_argument("--theorem4", action="store_true")
parser.add_argument("--epochs", default=1000,type=int)
parser.add_argument("--inner_dim",default=None,type=int)
parser.add_argument("--L",default=1.,type=float)

args = parser.parse_args()

# -----------------------------------------------------------
# 1.  Data
# -----------------------------------------------------------
sigma = 0.1
X, y = make_moons(n_samples=4000, noise=sigma, random_state=0)
X = StandardScaler().fit_transform(X)          # zero‑mean unit‑var
y = y.astype(np.float32).reshape(-1, 1)        # 0/1 → column vector

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BCEWithLogitsLoss expects float labels 0. or 1. and shape (N,1)
y_train = y_train.astype("float32").reshape(-1, 1)
y_val   = y_val.astype("float32").reshape(-1, 1)

# ----------------------------------------------------------------------
# 2.  Wrap into TensorDataset objects
# ----------------------------------------------------------------------
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float()
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).float()
)

train_ld = DataLoader(train_dataset, batch_size=256, shuffle=True,  drop_last=True)
val_ld   = DataLoader(val_dataset,   batch_size=512, shuffle=False, drop_last=False)

if args.inner_dim==None:
    dim_inner = len(X[0])+3
else:
    dim_inner = args.inner_dim
print("Inner dimension: ",dim_inner)

model = InvNet(n_blocks=args.layers,
               dim=2,          # input dim 2 !
               dim_inner=dim_inner,
               L=args.L,
               theorem4=args.theorem4)  # use the architecture for theorem 4 from the paper

model.to(device)
epochs = args.epochs
min_lr = 5e-3
max_lr = 1e-1
steps = int(epochs * len(X_train))
loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=min_lr)

scheduler = OneCycleLR(
        optim,
        max_lr=max_lr,
        div_factor= max_lr / min_lr,
        total_steps=steps,
        steps_per_epoch=len(train_ld),
        pct_start=0.5,
        anneal_strategy="cos",
        )

print_every = epochs // 20 if epochs > 20 else 1

train_model(
    model,
    train_ld,
    val_ld,
    optim=optim,
    scheduler=scheduler,
    epochs=epochs,
    device=device,
    loss_fn=loss_fn,
    print_every=print_every)