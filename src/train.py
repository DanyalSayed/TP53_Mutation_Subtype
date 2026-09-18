"""Training loop shared by notebooks/01_main_experiment.ipynb and
notebooks/02_window_ablation.ipynb.

Fixed hyperparameters (per experiment spec): Adam lr=1e-4, batch_size=128,
max_epochs=50, cross-entropy with inverse-frequency class weights, early
stopping on validation accuracy (patience=15), ReduceLROnPlateau on
validation loss (factor=0.5, patience=5), gradient clipping max_norm=1.0,
mixed precision (FP16) on CUDA.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data import CLASSES


def compute_class_weights(labels, num_classes=len(CLASSES), device='cpu'):
    """Inverse-frequency class weights (sklearn 'balanced' scheme) as a
    float32 tensor of shape (num_classes,), indexed per CLASSES order.
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model,
    train_dataset,
    val_dataset,
    device,
    max_epochs=50,
    batch_size=128,
    lr=1e-4,
    patience=15,
    lr_patience=5,
    lr_factor=0.5,
    grad_clip_norm=1.0,
    class_weights=None,
    use_amp=None,
    seed=42,
    verbose=True,
):
    """Train `model` in place; returns (model_with_best_weights, history).

    Early stopping and checkpoint selection are both driven by validation
    accuracy (the best epoch's weights are restored before returning).
    """
    torch.manual_seed(seed)

    if use_amp is None:
        use_amp = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             pin_memory=(device.type == 'cuda'))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'lr': []}
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = running_loss / n_train

        model.eval()
        val_loss_sum, n_val, n_correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                n_val += xb.size(0)
                n_correct += (logits.argmax(dim=1) == yb).sum().item()

        val_loss = val_loss_sum / n_val
        val_acc = n_correct / n_val
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = ' *' if improved else ''
            print(f"epoch {epoch:>3}/{max_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}{marker}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_accuracy improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    history['best_val_accuracy'] = best_val_acc
    history['best_epoch'] = int(np.argmax(history['val_accuracy'])) + 1
    history['stopped_epoch'] = epoch

    return model, history


@torch.no_grad()
def predict(model, dataset, device, batch_size=256):
    """Run inference; returns (predicted_labels, probabilities) as numpy
    arrays of shape (N,) and (N, num_classes) respectively (softmax applied).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    return preds, probs
