"""Visualization utilities for the transformer package."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple

def plot_attention_weights(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k] or [seq_len_q, seq_len_k]
        tokens: Optional list of token strings for axis labels
        title: Title for the plot
        layer_idx: Optional layer index for title
        head_idx: Optional attention head index for title
        cmap: Matplotlib colormap name
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Handle different input shapes
    if len(attention_weights.shape) == 4:  # [batch_size, num_heads, seq_len_q, seq_len_k]
        if head_idx is not None:
            attention_weights = attention_weights[0, head_idx]  # Take first batch, specified head
        else:
            attention_weights = attention_weights[0].mean(dim=0)  # Average over heads
    elif len(attention_weights.shape) == 3:  # [num_heads, seq_len_q, seq_len_k]
        if head_idx is not None:
            attention_weights = attention_weights[head_idx]
        else:
            attention_weights = attention_weights.mean(dim=0)
    
    # Convert to numpy for plotting
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
    
    # Set title
    plot_title = title
    if layer_idx is not None:
        plot_title += f" (Layer {layer_idx})"
    if head_idx is not None:
        plot_title += f" (Head {head_idx})"
    ax.set_title(plot_title)
    
    # Set axis labels
    if tokens is not None:
        # Set x-axis labels (key sequence)
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", rotation_mode="anchor")
        
        # Set y-axis labels (query sequence)
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens)
    
    # Add grid lines
    ax.set_xticks(np.arange(attention_weights.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(attention_weights.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Label axes
    ax.set_xlabel("Key Sequence")
    ax.set_ylabel("Query Sequence")
    
    # Tight layout
    fig.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig

def plot_embeddings_pca(
    embeddings: torch.Tensor,
    tokens: Optional[List[str]] = None,
    title: str = "Token Embeddings PCA",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot token embeddings using PCA for dimensionality reduction.
    
    Args:
        embeddings: Tensor of shape [seq_len, embed_dim] or [batch_size, seq_len, embed_dim]
        tokens: Optional list of token strings for labels
        title: Title for the plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    from sklearn.decomposition import PCA
    
    # Handle different input shapes
    if len(embeddings.shape) == 3:  # [batch_size, seq_len, embed_dim]
        embeddings = embeddings[0]  # Take first batch
    
    # Convert to numpy for PCA
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
    
    # Add labels if provided
    if tokens is not None:
        for i, token in enumerate(tokens):
            ax.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                       fontsize=12, alpha=0.8)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig

def plot_attention_heads(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_idx: Optional[int] = None,
    max_heads: int = 4,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple attention heads in a grid.
    
    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        tokens: Optional list of token strings for axis labels
        layer_idx: Optional layer index for title
        max_heads: Maximum number of heads to plot
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Take first batch
    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights[0]
    
    num_heads = attention_weights.shape[0]
    heads_to_plot = min(num_heads, max_heads)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(heads_to_plot)))
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Convert to numpy for plotting
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Plot each head
    for i in range(heads_to_plot):
        ax = axes[i]
        im = ax.imshow(attention_weights[i], cmap="viridis")
        
        # Set title
        head_title = f"Head {i}"
        if layer_idx is not None:
            head_title = f"Layer {layer_idx}, {head_title}"
        ax.set_title(head_title)
        
        # Set axis labels for the bottom-right plot
        if i == heads_to_plot - 1 or i == grid_size * grid_size - 1:
            if tokens is not None:
                # Only add labels to the last plot to avoid overcrowding
                ax.set_xticks(np.arange(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
                ax.set_yticks(np.arange(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(heads_to_plot, grid_size * grid_size):
        axes[i].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Set title
    main_title = "Attention Heads"
    if layer_idx is not None:
        main_title += f" for Layer {layer_idx}"
    fig.suptitle(main_title, fontsize=16)
    
    # Tight layout
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return fig