import math

import torch

from base import Transform, CompressionPipeline, flatten
from hadamard import RandomizedHadamard, padToPowerOf2
from quantization_constants import QuantizationType, get_all_quantization_constants_tensors
from random_p import RandomP


class EdenQuantization(Transform):
  def __init__(self, bits,
               q_type: QuantizationType = 'max_lloyd',
               device='cpu'):

    self.centroids, self.boundaries = get_all_quantization_constants_tensors(q_type, device)

    self.prng = torch.Generator(device=device)

    bits_frac, bits_low = math.modf(bits)
    if math.isclose(bits_frac, 0):
      self.fractional_bits = False
      self.bits = bits
    else:
      self.fractional_bits = True
      self.bits_low = bits_low
      self.bits_high = bits_low + 1
      self.bits_frac = bits_frac

  def forward(self, x):
    d = x.numel()
    normalized_x = x * math.sqrt(d) / l2(x)

    if self.fractional_bits:
      seed = self.prng.seed()
      mask = bernoulli_mask(x.shape, x.device, self.bits_frac, self.prng)

      x_low, x_high = mask_split(normalized_x, mask)

      assignments_low = torch.bucketize(x_low, self.boundaries[self.bits_low])
      assignments_high = torch.bucketize(x_high, self.boundaries[self.bits_high])

      assignments = (assignments_low, assignments_high)

      unscaled_centers_vec_low = torch.take(self.centroids[self.bits_low], assignments_low)
      unscaled_centers_vec_high = torch.take(self.centroids[self.bits_high], assignments_high)

      unscaled_centers_vec = mask_combine(unscaled_centers_vec_low, unscaled_centers_vec_high, mask)
      context = [seed, x.shape]
    else:
      assignments = torch.bucketize(normalized_x, self.boundaries[self.bits])
      unscaled_centers_vec = torch.take(self.centroids[self.bits], assignments)
      context = []

    scale = sum_squares(x) / (unscaled_centers_vec @ x)

    context.append(scale)
    return assignments, context

  def backward(self, assignments, context):
    if self.fractional_bits:
      seed, scale, original_shape = context
      assignments_low, assignments_high = assignments

      unscaled_centers_vec_low = torch.take(self.centroids[self.bits_low], assignments_low)
      unscaled_centers_vec_high = torch.take(self.centroids[self.bits_high], assignments_high)

      mask = bernoulli_mask(original_shape, assignments_low.device, self.bits_frac, self.prng.manual_seed(seed))
      unscaled_centers_vec = mask_combine(unscaled_centers_vec_low, unscaled_centers_vec_high, mask)
    else:
      scale = context[0]
      unscaled_centers_vec = torch.take(self.centroids[self.bits], assignments)

    # restore scale
    return scale * unscaled_centers_vec, None


def bernoulli_mask(shape, device, p, prng):
  return torch.empty(shape, dtype=torch.bool, device=device).bernoulli_(p=p, generator=prng)


def mask_split(x, mask):
  x0 = torch.masked_select(x, torch.logical_not(mask))
  x1 = torch.masked_select(x, mask)
  return x0, x1


def mask_combine(x0, x1, mask):
  x = torch.empty(mask.shape, dtype=x0.dtype, device=x0.device)
  x.masked_scatter_(torch.logical_not(mask), x0)
  x.masked_scatter_(mask, x1)

  return x


def sum_squares(x): return torch.sum(x ** 2)


def l2(x): return torch.sqrt(sum_squares(x))


def eden_builder(bits, q_type: QuantizationType = 'max_lloyd', device='cpu'):
  """

  Args:
    bits: A positive real bit rate
    q_type: Either 'max_lloyd' (Section 3) or 'ee' (Section 4.3)
    device: Torch device to use

  Returns:
    EDEN compression scheme instance
  """
  transforms = [flatten]
  if bits < 1:
    transforms += [RandomP(bits, device=device)]
    bits = 1
  transforms += [
    padToPowerOf2,
    RandomizedHadamard(device),
    EdenQuantization(bits, q_type, device),
  ]
  return CompressionPipeline(transforms)
