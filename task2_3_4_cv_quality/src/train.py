"""Training-side data plumbing for the CV quality task.

This module builds the reproducible train / val / test ``DataLoader``
foundation the rest of the task will consume. It applies the transforms
defined in :mod:`preprocess` and handles class imbalance on the
training split via a :class:`~torch.utils.data.WeightedRandomSampler`.

Typical usage::

    from task2_3_4_cv_quality.src.train import create_dataloaders

    bundle = create_dataloaders()
    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from task2_3_4_cv_quality.src.preprocess import (
    ClassBalanceReport,
    check_class_balance,
    get_transforms,
    load_config,
)


@dataclass
class DataLoaderBundle:
    """Container for the objects produced by :func:`create_dataloaders`."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: List[str]
    train_indices: List[int]
    val_indices: List[int]
    train_class_counts: Dict[str, int]
    balance_report: ClassBalanceReport


def _resolve_path(path_str: str) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    return path


def create_data_splits(
    train_dir: Path,
    val_split: float,
    seed: int,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> Tuple[Subset, Subset, List[str], List[int], List[int]]:
    """Build train / val ``Subset`` datasets over a shared index split.

    Two independent :class:`~torchvision.datasets.ImageFolder` instances
    are created over ``train_dir`` so the training and validation
    subsets can apply different transforms despite sharing samples.

    Parameters
    ----------
    train_dir:
        Directory passed to :class:`ImageFolder` (the ``Train`` folder).
    val_split:
        Fraction of samples held out for validation (``0 < val_split < 1``).
    seed:
        Seed used for the reproducible random permutation of indices.
    train_transform, val_transform:
        Transform pipelines applied by the train and validation subsets
        respectively.

    Returns
    -------
    (train_subset, val_subset, classes, train_indices, val_indices)
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"`val_split` must be in (0, 1), got {val_split}.")

    base_dataset = ImageFolder(str(train_dir))
    n_total = len(base_dataset)
    if n_total == 0:
        raise RuntimeError(f"No images found under {train_dir}.")

    generator = torch.Generator().manual_seed(int(seed))
    permuted = torch.randperm(n_total, generator=generator).tolist()
    n_val = int(round(n_total * val_split))
    val_indices = permuted[:n_val]
    train_indices = permuted[n_val:]

    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(train_dir), transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset, list(base_dataset.classes), train_indices, val_indices


def get_training_class_counts(
    targets: Sequence[int],
    indices: Sequence[int],
    classes: Sequence[str],
) -> Dict[str, int]:
    """Return per-class counts restricted to ``indices``.

    Parameters
    ----------
    targets:
        Full list of integer labels (e.g. ``ImageFolder.targets``).
    indices:
        Indices of the training subset within ``targets``.
    classes:
        Ordered class names so the dict preserves ``ImageFolder`` order.
    """
    counter = Counter(int(targets[i]) for i in indices)
    return {name: int(counter.get(idx, 0)) for idx, name in enumerate(classes)}


def create_weighted_sampler(
    targets: Sequence[int],
    indices: Sequence[int],
    num_classes: int,
) -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` that oversamples minorities.

    Each training sample receives a weight of ``1 / count[class]`` so
    rare classes are drawn more often. Sampling is performed with
    replacement and for one full training-subset-worth of samples per
    epoch.
    """
    class_counts = [0] * num_classes
    for i in indices:
        class_counts[int(targets[i])] += 1

    class_weights = [
        1.0 / c if c > 0 else 0.0 for c in class_counts
    ]
    sample_weights = torch.as_tensor(
        [class_weights[int(targets[i])] for i in indices],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True,
    )


def create_dataloaders(
    config: Optional[dict] = None,
    verbose: bool = True,
) -> DataLoaderBundle:
    """Build train / val / test ``DataLoader`` objects for the task.

    The training loader uses a :class:`WeightedRandomSampler` (no
    ``shuffle=True``) to oversample minority classes; the validation
    and test loaders are plain, unshuffled loaders with deterministic
    transforms.
    """
    cfg = config if config is not None else load_config()

    train_dir = _resolve_path(cfg["train_dir"])
    test_dir = _resolve_path(cfg["test_dir"])
    val_split = float(cfg["val_split"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    seed = int(cfg["random_seed"])

    train_tf = get_transforms("train")
    val_tf = get_transforms("val")
    test_tf = get_transforms("test")

    train_subset, val_subset, classes, train_indices, val_indices = create_data_splits(
        train_dir=train_dir,
        val_split=val_split,
        seed=seed,
        train_transform=train_tf,
        val_transform=val_tf,
    )

    base_targets = train_subset.dataset.targets

    train_class_counts = get_training_class_counts(
        targets=base_targets, indices=train_indices, classes=classes
    )
    balance_report = check_class_balance(
        train_class_counts, verbose=verbose
    )

    sampler = create_weighted_sampler(
        targets=base_targets,
        indices=train_indices,
        num_classes=len(classes),
    )

    test_dataset = ImageFolder(str(test_dir), transform=test_tf)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if verbose:
        print()
        print(f"Classes ({len(classes)}): {classes}")
        print(f"Train samples : {len(train_subset)}")
        print(f"Val   samples : {len(val_subset)}")
        print(f"Test  samples : {len(test_dataset)}")

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        train_indices=train_indices,
        val_indices=val_indices,
        train_class_counts=train_class_counts,
        balance_report=balance_report,
    )


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

import json
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm.auto import tqdm


def train_one_epoch_standalone(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number (for display)
    
    Returns
    -------
    tuple[float, float]
        (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "Val",
) -> tuple[float, float]:
    """Evaluate the model.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        Data loader (validation or test)
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to evaluate on
    split_name : str
        Name of split for display ("Val" or "Test")
    
    Returns
    -------
    tuple[float, float]
        (average_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"{split_name:>4}", leave=False)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model_type: str = "custom_cnn",
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    step_size: int = 10,
    gamma: float = 0.5,
    config: Optional[Dict] = None,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Complete training pipeline.
    
    Parameters
    ----------
    model_type : str
        Model architecture ("custom_cnn", "resnet50", "efficientnet_b0")
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization strength
    step_size : int
        StepLR scheduler step size
    gamma : float
        StepLR scheduler gamma (LR *= gamma every step_size epochs)
    config : dict, optional
        Config override (if None, loads from config.yaml)
    save_dir : Path, optional
        Directory to save models (default: models/)
    
    Returns
    -------
    dict
        Training history and final metrics
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if save_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        save_dir = repo_root / "task2_3_4_cv_quality" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    bundle = create_dataloaders(config=config, verbose=True)
    
    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader
    classes = bundle.classes
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    from task2_3_4_cv_quality.src.model import get_model, print_model_summary, count_parameters
    
    model = get_model(
        num_classes=len(classes),
        model_type=model_type,
        pretrained=(model_type != "custom_cnn"),
        dropout_rate=0.5
    ).to(device)
    
    print_model_summary(model, model_type=model_type, verbose=False)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    
    print(f"\nOptimizer:  Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Scheduler:  StepLR (step_size={step_size}, gamma={gamma})")
    print(f"Loss:       CrossEntropyLoss")
    
    # Training loop
    print("\n" + "="*80)
    print(f"TRAINING: {model_type.upper()} FOR {num_epochs} EPOCHS")
    print("="*80)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_one_epoch_standalone(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, split_name="Val"
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR:    {current_lr:.6f}")
        
        # Save best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'num_classes': len(classes),
                'classes': classes,
            }
            
            best_model_path = save_dir / f"{model_type}_best.pt"
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Saved new best model (val_acc: {val_acc:.2f}%) → {best_model_path.name}")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    best_checkpoint = torch.load(save_dir / f"{model_type}_best.pt")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, split_name="Test"
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Acc:  {test_acc:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_checkpoint['epoch']})")
    print(f"Test Accuracy:            {test_acc:.2f}%")
    print(f"Model saved to:           {best_model_path}")
    print("="*80)
    
    # AA-38: Save metadata JSON alongside the model
    metadata = {
        "model_type": model_type,
        "epochs_trained": num_epochs,
        "best_epoch": best_checkpoint['epoch'],
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_classes": len(classes),
        "classes": classes,
        "date_trained": datetime.now().isoformat(),
    }
    metadata_path = save_dir / f"{model_type}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_epoch': best_checkpoint['epoch'],
        'model_path': str(best_model_path),
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    """Main entry point for command-line training."""
    import argparse

    # Load config defaults so CLI flags override them
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Train CV quality grading model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', type=str,
        default=cfg.get("model_type", "custom_cnn"),
        choices=['custom_cnn', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=cfg.get("epochs", 20),
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int,
        default=cfg.get("batch_size", 32),
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float,
        default=cfg.get("learning_rate", 0.001),
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight-decay', type=float,
        default=cfg.get("weight_decay", 0.0001),
        help='L2 regularization strength'
    )
    parser.add_argument(
        '--step-size', type=int,
        default=cfg.get("step_size", 10),
        help='StepLR scheduler step size'
    )
    parser.add_argument(
        '--gamma', type=float,
        default=cfg.get("gamma", 0.5),
        help='StepLR scheduler gamma'
    )
    parser.add_argument(
        '--save-dir', type=str,
        default=None,
        help='Directory to save models (default: models/)'
    )

    args = parser.parse_args()

    # Override batch_size in config so create_dataloaders picks it up
    cfg["batch_size"] = args.batch_size

    print("=" * 80)
    print("CV QUALITY GRADING - MODEL TRAINING")
    print("=" * 80)
    print(f"Model:        {args.model}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Scheduler:    StepLR (step={args.step_size}, gamma={args.gamma})")
    print("=" * 80)

    save_dir = Path(args.save_dir) if args.save_dir else None

    results = train_model(
        model_type=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        config=cfg,
        save_dir=save_dir,
    )

    print("\n✓ Training completed successfully!")
    print(f"✓ Best model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()