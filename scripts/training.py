import torch
import numpy as np
import torch.nn as nn
import os
import time

def to_tensor(a, device="cpu"):
    return torch.from_numpy(a).float().to(device)

def train_model(
    model,
    train_ld,
    val_ld,
    optim,
    scheduler=None,
    epochs=1000,
    device="cpu",
    loss_fn=nn.BCEWithLogitsLoss(),
    print_every=50):
    
    rng = np.random.default_rng(0)
    
    istheorem4 = "theorem4" if model.theorem4 else "theorem3"
    
    if not os.path.exists(f"saved_results"):
        os.makedirs(f"saved_results")
    
    with open(f"saved_results/{istheorem4}_results.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + model.string_descrition + "\n")  
    
    avg_normalisation_per_epoch = 0.
    
    for ep in range(1, epochs + 1):
        
        model.train()
        running_loss = 0.0
        time_normalisation = 0.
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)         # shape B×1  (raw scores)
            loss   = loss_fn(logits, yb)
            running_loss += loss.item() * xb.size(0)

            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            start_normalisation = time.time()
            
            #Compute one step of the power method
            model.update_spectral_norms()
            
            #Clip the final scaling factor
            model.clip_scaling()
            
            with torch.no_grad():
                if model.theorem4:
                    model.normalise_lifting_upper()
                    model.normalise_affines()
            
            w_last = model.last.weight.data
            model.last.weight.data = w_last / torch.maximum(torch.sum(torch.abs(w_last)), torch.tensor(1.0,device=w_last.device))
            end_normalisation = time.time()
            time_normalisation += end_normalisation - start_normalisation
        
        avg_normalisation_per_epoch += time_normalisation
        
        if ep % print_every == 0 or ep == 1:
            model.eval()
            val_loss = 0.0
            n_val    = 0
            n_correct= 0
            with torch.no_grad():
                for xv, yv in val_ld:
                    xv, yv = xv.to(device), yv.to(device)
                    logits = model(xv)
                    val_loss += loss_fn(logits, yv).item() * xv.size(0)
                    preds = (logits > 0).float()
                    n_correct += (preds == yv).sum().item()
                    n_val     += xv.size(0)

            output = f"epoch {ep:3d}/{epochs} | " \
                  f"train loss {(running_loss/len(train_ld.dataset)):.4f} | " \
                  f"val loss {(val_loss/n_val):.4f} | " \
                  f"val acc {100*n_correct/n_val:.4f}% | " \
                  f"normalisation time epoch: {time_normalisation:.4f}"
            print(output)
            
            with open(f"saved_results/{istheorem4}_results.txt", "a", encoding="utf-8") as fh:
                fh.write(output + "\n")
    
    if not os.path.exists(f"saved_models"):
        os.makedirs(f"saved_models")
    torch.save(model.state_dict(), f"saved_models/{istheorem4}_layers_{model.n_blocks}.pt")
    
    
    with open(f"saved_results/{istheorem4}_results.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Average normalisation time per epoch {avg_normalisation_per_epoch/epochs:.4f}" + "\n")

    
def train_model_MNIST(
    model,
    train_ld,
    val_ld,
    optim,
    scheduler=None,
    epochs: int = 100,
    device: str = "cpu",
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    print_every: int = 10,
):

    istheorem4 = "theorem4" if model.backbone.theorem4 else "theorem3"
    os.makedirs("saved_models", exist_ok=True)

    if not os.path.exists(f"saved_results"):
        os.makedirs(f"saved_results")
        
    with open(f"saved_results/{istheorem4}_results_MNIST.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + model.backbone.string_descrition + "\n")

    avg_norm_time_per_epoch = 0.0

    for ep in range(1, epochs + 1):

        model.train()
        running_loss = 0.0
        norm_time_epoch = 0.0

        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device, dtype=torch.long)  # int labels
            optim.zero_grad()

            logits = model(xb)                         # [B, K]
            loss   = loss_fn(logits, yb)               # CE expects ints
            running_loss += loss.item() * xb.size(0)

            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            # ── 1-Lipschitz normalisation + projections ────────────────────────
            t0 = time.time()

            model.backbone.update_spectral_norms()
            model.backbone.clip_scaling()

            with torch.no_grad():
                if model.backbone.theorem4:
                    model.backbone.normalise_affines()
            norm_time_epoch += time.time() - t0

        avg_norm_time_per_epoch += norm_time_epoch

        if ep % print_every == 0 or ep == 1:
            model.eval()
            val_loss, n_val, n_correct = 0.0, 0, 0
            with torch.no_grad():
                for xv, yv in val_ld:
                    xv, yv = xv.to(device), yv.to(device, dtype=torch.long)
                    logits = model(xv)
                    val_loss += loss_fn(logits, yv).item() * xv.size(0)

                    preds = logits.argmax(dim=1)            # top-1 prediction
                    n_correct += (preds == yv).sum().item()
                    n_val     += xv.size(0)

            print(
                f"epoch {ep:3d}/{epochs} | "
                f"train loss {(running_loss/len(train_ld.dataset)):.4f} | "
                f"val loss {(val_loss/n_val):.4f} | "
                f"val acc {100*n_correct/n_val:.2f}% | "
                f"normalisation time {norm_time_epoch:.3f}s"
            )

            with open(f"saved_results/{istheorem4}_results_MNIST.txt", "a", encoding="utf-8") as fh:
                fh.write(
                    f"epoch {ep:3d}/{epochs} | "
                    f"train loss {(running_loss/len(train_ld.dataset)):.4f} | "
                    f"val loss {(val_loss/n_val):.4f} | "
                    f"val acc {100*n_correct/n_val:.2f}% | "
                    f"normalisation time {norm_time_epoch:.3f}s\n"
                )

    # ── SAVE ───────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), f"saved_models/{istheorem4}_layers_{model.backbone.n_blocks}_hidden_{model.backbone.dim}_MNIST.pt")

    with open(f"saved_results/{istheorem4}_results_MNIST.txt", "a", encoding="utf-8") as fh:
        fh.write(f"Average normalisation time/epoch: {avg_norm_time_per_epoch/epochs:.3f}s\n")
