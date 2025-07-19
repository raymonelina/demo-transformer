"""Debug utilities for the transformer package."""

import torch
from typing import Optional, Union, List, Tuple


def debug_print(
    tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    name: str,
    purpose: str = "",
    prefix: str = "",
    print_values: bool = False,
    max_elements: int = 10
) -> None:
    """
    Print debug information about a tensor or list of tensors.
    
    Args:
        tensor: The tensor or list of tensors to debug print
        name: Name of the tensor for identification
        purpose: Description of what this tensor represents
        prefix: Prefix for the debug message (e.g., module name)
        print_values: Whether to print actual tensor values (limited by max_elements)
        max_elements: Maximum number of elements to print if print_values is True
    """
    if isinstance(tensor, (list, tuple)):
        print(f"{prefix}[DEBUG] {name} (List/Tuple of {len(tensor)} tensors): {purpose}")
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                shape_str = f"shape={tuple(t.shape)}"
                dtype_str = f"dtype={t.dtype}"
                device_str = f"device={t.device}"
                print(f"{prefix}  - Item {i}: {shape_str}, {dtype_str}, {device_str}")
                if print_values and t.numel() <= max_elements:
                    print(f"{prefix}    Values: {t.detach().cpu().tolist()}")
                elif print_values:
                    flat = t.detach().cpu().flatten()
                    print(f"{prefix}    First {max_elements} values: {flat[:max_elements].tolist()}")
                    print(f"{prefix}    Min: {t.min().item()}, Max: {t.max().item()}, Mean: {t.mean().item()}")
    elif isinstance(tensor, torch.Tensor):
        shape_str = f"shape={tuple(tensor.shape)}"
        dtype_str = f"dtype={tensor.dtype}"
        device_str = f"device={tensor.device}"
        print(f"{prefix}[DEBUG] {name}: {shape_str}, {dtype_str}, {device_str} - {purpose}")
        
        if print_values and tensor.numel() <= max_elements:
            print(f"{prefix}  Values: {tensor.detach().cpu().tolist()}")
        elif print_values:
            flat = tensor.detach().cpu().flatten()
            print(f"{prefix}  First {max_elements} values: {flat[:max_elements].tolist()}")
            print(f"{prefix}  Min: {tensor.min().item()}, Max: {tensor.max().item()}, Mean: {tensor.mean().item()}")
    else:
        print(f"{prefix}[DEBUG] {name} (Not a tensor): {tensor}")