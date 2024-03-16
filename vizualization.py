from dataloader import DataLoader, Data
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Callable

def show_sample_outputs_from_dataloader(data: Data):
    data.train.ShowImage()
    data.test.ShowRotation()
    data.val.ShowTransform()

    print(f'Images shape: {data.train.images.shape}')
    print(f'Poses shape: {data.train.transforms.shape}')

def visualize_poses_in_3d(data: Data):
    dirs = np.array([np.sum([0, 0, -1] * transform[:3, :3], axis=-1) for transform in data.train.transforms])
    origins = data.train.transforms[:, :3, -1]

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        dirs[:, 0],
        dirs[:, 1],
        dirs[:, 2],
        length=0.5, normalize=True
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.show()

def plot_stratified_sampling(ts: torch.Tensor):
    y_vals = torch.zeros_like(ts)

    plt.plot(ts[0].cpu().numpy(), y_vals[0].cpu().numpy(), 'r-o')
    plt.ylim([-1, 2])
    plt.title('Stratified Sampling')
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
    plt.show()

def show_sample_outputs_from_positional_encoding(
    x_encoded: torch.Tensor,
    d_encoded: torch.Tensor
):
    print('Encoded Points - shape:')
    print(x_encoded.shape)
    print('Encoded Points - min, max, mean values:')
    print(torch.min(x_encoded), torch.max(x_encoded), torch.mean(x_encoded))
    print('')

    print('Encoded Directions - shape:')
    print(d_encoded.shape)
    print('Encoded Directions - min, max, mean values:')
    print(torch.min(d_encoded), torch.max(d_encoded), torch.mean(d_encoded))
    print('')

def plot_samples(
  z_vals: torch.Tensor,
  z_hierarch: Optional[torch.Tensor] = None,
  ax: Optional[np.ndarray] = None):
  """
  Plot stratified and (optional) hierarchical samples.
  """
  y_vals = 1 + np.zeros_like(z_vals)

  if ax is None:
    ax = plt.subplot()
  ax.plot(z_vals, y_vals, 'b-o')
  if z_hierarch is not None:
    y_hierarch = np.zeros_like(z_hierarch)
    ax.plot(z_hierarch, y_hierarch, 'r-o')
  ax.set_ylim([-1, 2])
  ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
  ax.axes.yaxis.set_visible(False)
  ax.grid(True)
  return ax