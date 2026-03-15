import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    return DEVICE


def to_device(x):
    """Move tensor or array to the configured device."""
    if hasattr(x, "to"):
        return x.to(DEVICE)
    return x


def move_tn_to_device(tn):
    """Move a quimb TensorNetwork to the configured device."""
    tn.apply_to_arrays(to_device)
    return tn


def move_data_to_device(data: dict) -> dict:
    """Move all tensors in a data dict to the configured device."""
    return {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in data.items()}
