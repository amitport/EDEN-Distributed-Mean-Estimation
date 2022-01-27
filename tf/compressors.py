from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.ops import math_ops

from distributed_dp import compression_utils
from functools import partial
from absl import logging
import tensorflow_federated as tff

from tf.eden import eden_quantization, inverse_eden_quantization
from tf.hadamard import _create_hadamard_fn

SUPPORTED_COMPRESSORS = ['noop', 'eden', 'hadamard', 'kashin', 'sq']


def sample_indices(shape, p, seed):
  mask = tf.less(tf.random.stateless_uniform(shape, seed=seed), p)
  return tf.where(mask)


def eden_roundtrip(input_record,
                   hadamard_seed: tf.Tensor,
                   rand_p_seed: tf.Tensor,
                   p=0.1,
                   bits=1):
  """Applies compression to the record as a single concatenated vector."""
  input_vec = compression_utils.flatten_concat(input_record)

  casted_record = tf.cast(input_vec, tf.float32)

  rotated_record = compression_utils.randomized_hadamard_transform(
    casted_record, seed_pair=hadamard_seed)

  sparse_indices = sample_indices(rotated_record.shape, p, rand_p_seed)
  sparse_record = tf.reshape(tf.gather(rotated_record, sparse_indices) * (1 / p), [-1])

  quantized_record, scale = eden_quantization(sparse_record, bits=bits)
  dequantized_record = inverse_eden_quantization(quantized_record, scale,
                                                 bits=bits)

  desparse_record = tf.scatter_nd(
    sparse_indices,
    dequantized_record,
    tf.cast(rotated_record.shape, sparse_indices.dtype)
  )

  unrotated_record = compression_utils.inverse_randomized_hadamard_transform(
    desparse_record,
    original_dim=tf.size(casted_record),
    seed_pair=hadamard_seed)

  if input_vec.dtype.is_integer:
    uncasted_input_vec = tf.cast(tf.round(unrotated_record), input_vec.dtype)
  else:
    uncasted_input_vec = tf.cast(unrotated_record, input_vec.dtype)

  reconstructed_record = compression_utils.inverse_flatten_concat(
    uncasted_input_vec, input_record)
  return reconstructed_record


def _create_eden_fn(value_type, bits, p):
  @tff.tf_computation(value_type)
  def eden_fn(record):
    microseconds_per_second = 10 ** 6  # Timestamp returns fractional seconds.
    timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                     tf.int32)
    hadamard_seed = tf.convert_to_tensor([timestamp_microseconds, 0])

    rand_p_seed = tf.convert_to_tensor([timestamp_microseconds * 2, 0])

    return eden_roundtrip(record, hadamard_seed=hadamard_seed, rand_p_seed=rand_p_seed, p=p, bits=bits)

  return eden_fn


def get_compressor_factory(compressor: str, **kwargs):
  logging.info(f'compressor {compressor}!')
  logging.info(kwargs)

  if compressor == 'noop':
    def _create_noop_fn(value_type):
      @tff.tf_computation(value_type)
      def noop_fn(record):
        return record

      return noop_fn

    return _create_noop_fn
  else:
    bits = kwargs['num_bits']
    p = kwargs['p']

    if compressor == 'eden':
      _roundtrip_fn = _create_eden_fn
    elif compressor == 'hadamard':
      _roundtrip_fn = _create_hadamard_fn
    elif compressor == 'kashin':
      _roundtrip_fn = _create_kashin_fn
    elif compressor == 'sq':
      _roundtrip_fn = _create_sq_fn
    else:
      raise ValueError('expected a compressor name')

    return partial(_roundtrip_fn, bits=bits, p=p)
