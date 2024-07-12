import torch
import torch.nn.functional as F


def create_hierarchical_labels(x):
    """
    Create hierarchical labels from a one-hot or multi-hot encoded vector.

    This function takes a vector where '1' indicates a positive label and transforms it
    into a hierarchical representation where:
    - Original positive labels (1s) become 2s
    - Positions immediately next to 2s become 1s
    - All other positions remain 0s

    Args:
    x (torch.Tensor): Input tensor with one-hot or multi-hot encoding

    Returns:
    torch.Tensor: Hierarchical label tensor
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    # Create a copy of the input tensor to avoid modifying the original
    hierarchical = x.clone().float()

    # Step 1: Replace all 1s with 2s
    # This elevates the importance of the original positive labels
    hierarchical[hierarchical == 1] = 2

    # Step 2: Find positions of all 2s
    # These are the locations of our original positive labels
    positions = torch.where(hierarchical == 2)[0]

    # Step 3: Add 1s next to 2s
    for pos in positions:
        # Check left neighbor (if it exists)
        if pos > 0:
            # Set to 1, but don't overwrite any existing 2
            hierarchical[pos - 1] = max(hierarchical[pos - 1], 1)

        # Check right neighbor (if it exists)
        if pos < len(hierarchical) - 1:
            # Set to 1, but don't overwrite any existing 2
            hierarchical[pos + 1] = max(hierarchical[pos + 1], 1)

    return hierarchical


def HierarchicalPartialLoss(output, target, axis=-1):
    """
    Compute the Hierarchical Partial Loss.

    This loss function is designed for scenarios where there's a hierarchical relationship
    between classes, and partial credit should be given for predictions that are "close"
    to the correct class.

    Args:
    output (torch.Tensor): The raw output (logits) from the model
    target (torch.Tensor): The target labels (one-hot or multi-hot encoded)
    axis (int): The axis along which to compute the loss (default is -1, the last dimension)

    Returns:
    torch.Tensor: The computed loss value
    """

    # Step 1: Create hierarchical labels from the target
    # This transforms our original labels into a hierarchical representation
    target = create_hierarchical_labels(target)

    # Step 2: Define epsilon (a small value to prevent division by zero or log of zero)
    # We use the smallest positive value representable by the data type of 'output'
    epsilon = torch.finfo(output.dtype).eps
    epsilon_ = torch.tensor(epsilon, dtype=output.dtype)

    # Step 3: Normalize the output (convert raw logits to probabilities)
    # We add epsilon to the denominator to prevent division by zero
    output = output / (torch.sum(output, dim=axis, keepdim=True) + epsilon)

    # Step 4: Create wide and narrow masks from the hierarchical target
    # Wide mask: Considers both exact matches (2s) and near matches (1s)
    # Narrow mask: Considers only exact matches (2s)
    mask_wide = torch.clamp(target, 0, 1)  # Transforms 2s to 1s, keeps 1s and 0s
    mask_narrow = torch.clamp(target, 1, 2) - 1  # Transforms 2s to 1s, 1s and 0s to 0s

    # Step 5: Compute the wide loss (considers both exact and near matches)
    # We sum the product of the wide mask and output, then take the negative log
    # Clipping is used to prevent log(0) which is undefined
    loss_wide = -torch.log(torch.clamp(
        torch.sum(mask_wide * output, dim=axis),
        min=epsilon_,
        max=1. - epsilon_
    ))

    # Step 6: Compute the narrow loss (considers only exact matches)
    # Similar to wide loss, but uses the narrow mask
    loss_narrow = -torch.log(torch.clamp(
        torch.sum(mask_narrow * output, dim=axis),
        min=epsilon_,
        max=1. - epsilon_
    ))

    # Step 7: Return the sum of wide and narrow losses
    # This combines the penalties for both near misses and exact misses
    return loss_wide + loss_narrow
