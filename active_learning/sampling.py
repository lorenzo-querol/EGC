import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from guided_diffusion import dist_util


def random_sampling(u_set: np.ndarray, budget_size: int, seed: int = 42):
    """
    Randomly sample points from unlabeled set.

    Args:
        u_set: Indices of unlabeled samples
        budget_size: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        tuple: (active_set, remain_set)
            - active_set: Selected sample indices
            - remain_set: Remaining unlabeled indices

    Raises:
        TypeError: If inputs are not of correct type
        ValueError: If budget_size is invalid
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Validate inputs
    if not isinstance(u_set, np.ndarray):
        raise TypeError(f"u_set must be numpy.ndarray, got {type(u_set)}")
    if not isinstance(budget_size, int):
        raise TypeError(f"budget_size must be int, got {type(budget_size)}")
    if budget_size <= 0:
        raise ValueError("budget_size must be positive")
    if budget_size >= len(u_set):
        raise ValueError(f"budget_size ({budget_size}) cannot exceed unlabeled set size ({len(u_set)})")

    # Generate random indices
    indices = np.arange(len(u_set))
    np.random.shuffle(indices)

    # Split into active and remaining sets
    active_set = u_set[indices[:budget_size]]
    remain_set = u_set[indices[budget_size:]]

    return active_set, remain_set


def uncertainty_sampling(model, unlabeled_dataloader, u_set, budget_size: int):
    """
    Sample points using uncertainty principle as acquisition function.

    Args:
        model: PyTorch model in eval mode
        unlabeled_dataloader: DataLoader for unlabeled samples
        budget_size: Number of samples to select

    Returns:
        tuple: (active_set, remain_set)
            - active_set: Selected sample indices
            - remain_set: Remaining unlabeled indices

    Raises:
        ValueError: If model is in training mode
    """
    if model.training:
        raise ValueError("Model must be in eval mode")

    # Move model to GPU
    model = model.to(dist_util.dev())

    # Compute uncertainty scores
    uncertainty_scores = []
    for images, _ in tqdm(unlabeled_dataloader, desc="Computing uncertainty scores"):
        with torch.no_grad():
            images = images.to(dist_util.dev())

            # Get softmax probabilities and compute uncertainty as 1 - max_prob
            probs = F.softmax(model(images), dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            uncertainty = 1 - max_probs
            uncertainty_scores.append(uncertainty.cpu().numpy())

    # Combine scores from all batches
    uncertainty_scores = np.concatenate(uncertainty_scores, axis=0)

    # Sort by uncertainty (highest to lowest)
    sorted_indices = np.argsort(uncertainty_scores)[::-1]

    # Select samples
    active_set = u_set[sorted_indices[:budget_size]]
    remain_set = u_set[sorted_indices[budget_size:]]

    return active_set, remain_set
