from math import floor, log2

import torch
import torch.nn.functional as F

from base import Transform


def hadamard_transform_(vec):
  """fast Walsh–Hadamard transform (in-place)

  :param vec: vec is expected to be a power of 2!
  :return: the Hadamard transform of vec
  """
  d = vec.numel()
  original_shape = vec.shape
  h = 2
  while h <= d:
    hf = h // 2
    vec = vec.view(d // h, h)

    ## the following is a more inplace way of doing the following:
    # half_1 = batch[:, :, :hf]
    # half_2 = batch[:, :, hf:]
    # batch = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
    # the NOT inplace seems to be actually be slightly faster
    # (I assume for making more memory-contiguous operations. That being said,
    # it more easily throws out-of-memory and may slow things overall,
    # so using inplace version below:)

    vec[:, :hf] = vec[:, :hf] + vec[:, hf:2 * hf]
    vec[:, hf:2 * hf] = vec[:, :hf] - 2 * vec[:, hf:2 * hf]
    h *= 2

  vec *= d ** -0.5  # vec /= np.sqrt(d)

  return vec.view(*original_shape)


def rademacher_like(x, generator):
  """ (previously random_diagonal) """
  return 2 * torch.torch.empty_like(x).bernoulli_(generator=generator) - 1


def randomized_hadamard_transform_(x, generator):
  d = rademacher_like(x, generator)

  return hadamard_transform_(x * d)


def inverse_randomized_hadamard_transform_(tx, generator):
  d = rademacher_like(tx, generator)

  return hadamard_transform_(tx) * d


class RandomizedHadamard(Transform):
  """
    Assumes that the input is a vector and a power of 2
  """

  def __init__(self, device='cpu'):
    self.prng = torch.Generator(device=device)

  def forward(self, x):
    seed = self.prng.seed()

    return randomized_hadamard_transform_(x, self.prng), seed

  def backward(self, tx, seed):
    return inverse_randomized_hadamard_transform_(tx, self.prng.manual_seed(seed)), None


def next_power_of_2(n):
  return 2 ** (floor(log2(n)) + 1)


class PadToPowerOf2(Transform):
  def forward(self, x):
    """_
    :param x: assumes vec is 1d
    :return: x padded with zero until the next power-of-2
    """
    d = x.numel()
    # pad to the nearest power of 2 if needed
    if d & (d - 1) != 0:
      dim_with_pad = next_power_of_2(d)
      x = F.pad(x, (0, dim_with_pad - d))

    return x, d

  def backward(self, tx, original_dim):
    return tx[:original_dim], None
