import torch

from base import Transform


class RandomP(Transform):
  """
  Random Sparsification given a parameter p remaining to determine the
  probability of a coordinate to keep
  """

  def __init__(self, p=0.5, device='cpu'):
    self.prng = torch.Generator(device=device)
    self.p = p

  def forward(self, x):
    seed = self.prng.seed()
    original_shape = x.shape
    original_d = x.numel()
    mask = torch.empty_like(x).bernoulli_(p=self.p, generator=self.prng)

    indices = torch.nonzero(mask, as_tuple=True)

    scale = 1 / self.p
    return x[indices] * scale, (seed, original_shape, original_d)

  def backward(self, sparse_x, context):
    seed, original_shape, original_d = context

    x = torch.zeros(original_d, dtype=sparse_x.dtype, layout=sparse_x.layout,
                    device=sparse_x.device)

    indices = torch.nonzero(
      torch.empty_like(x).bernoulli_(p=self.p, generator=self.prng.manual_seed(seed))
    ).squeeze()
    x.scatter_(0, indices, sparse_x)

    return x.view(original_shape), None
