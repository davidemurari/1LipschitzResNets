import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms

from scripts.training import train_model_MNIST 
from scripts.networks import InvNet, InvNetClassifier

# -----------------------------------------------------------------------------
# 0.  Command‑line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MNIST classification with InvNet")
parser.add_argument("--layers",      default=5,   type=int,   help="Number of invertible blocks")
parser.add_argument("--inner_dim",   default=100,type=int,   help="Hidden width of each block (default: input_dim + 3)")
parser.add_argument("--L",           default=1.0, type=float, help="Spectral normalisation constant (see paper)")
parser.add_argument("--epochs",      default=100,  type=int,   help="Training epochs")
parser.add_argument("--batch_size",  default=256, type=int,   help="Mini‑batch size")
parser.add_argument("--lr_min",      default=5e-4,type=float, help="Lower LR bound for 1‑cycle")
parser.add_argument("--lr_max",      default=1e-2,type=float, help="Upper LR bound for 1‑cycle")
parser.add_argument("--theorem4",    action="store_true",    help="Use the Theorem‑4 architecture variant")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 1.  Data
# -----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                 # (0,1] floats
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
val_ds   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)#, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False)#, pin_memory=True)

# -----------------------------------------------------------------------------
# 2.  Model – InvNet feature extractor + linear classifier head
# -----------------------------------------------------------------------------
INPUT_DIM   = 28 * 28          # MNIST images flattened
NUM_CLASSES = 10

if args.inner_dim is None:
    dim_inner = INPUT_DIM + 3
else:
    dim_inner = args.inner_dim

model = InvNetClassifier(
    n_blocks=args.layers,
    first_dim=args.inner_dim,
    L=args.L,
    theorem4=args.theorem4,
    n_classes=NUM_CLASSES,
    INPUT_DIM=INPUT_DIM
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------------------------------------------------------
# 3.  Optimiser, scheduler, and loss
# -----------------------------------------------------------------------------
optim = torch.optim.Adam(model.parameters(), lr=args.lr_min)
steps_per_epoch = len(train_ld)
total_steps = args.epochs * steps_per_epoch

scheduler = OneCycleLR(
    optim,
    max_lr=args.lr_max,
    div_factor=args.lr_max / args.lr_min,
    total_steps=total_steps,
    anneal_strategy="cos",
    pct_start=0.5,
)

criterion = nn.CrossEntropyLoss()

epochs = args.epochs

print_every = epochs // 20 if epochs > 20 else 1

train_model_MNIST(
    model,
    train_ld,
    val_ld,
    optim=optim,
    scheduler=scheduler,
    epochs=epochs,
    device=device,
    loss_fn=criterion,
    print_every=print_every)
