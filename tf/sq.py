import tensorflow as tf
import tensorflow_federated as tff
from distributed_dp import compression_utils


def sample_indices(shape, p, seed):
  mask = tf.less(tf.random.stateless_uniform(shape, seed=seed), p)
  return tf.where(mask)


def round_fn(x):
  floored_x = tf.floor(x)
  decimal_x = x - floored_x

  bernoulli = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=0, maxval=1) < decimal_x
  rounded_x = floored_x + tf.cast(bernoulli, x.dtype)

  return rounded_x


def sq_roundtrip(input_record,
                 rand_p_seed: tf.Tensor,
                 p=0.1,
                 bits=1):
  """Applies compression to the record as a single concatenated vector."""
  x = compression_utils.flatten_concat(input_record)
  flat_shape = x.shape

  if p < 1:
    sparse_indices = sample_indices(x.shape, p, rand_p_seed)
    x = tf.reshape(
      tf.gather(x, sparse_indices) * (1 / p),
      [-1])

  max_int_value = tf.cast(2 ** bits - 1, x.dtype)
  offset, x_max = tf.reduce_min(x), tf.reduce_max(x)
  scale = x_max - offset
  x = round_fn(tf.math.divide_no_nan(x - offset, scale) * max_int_value)
  x = x / max_int_value * scale + offset

  if p < 1:
    x = tf.scatter_nd(
      sparse_indices,
      x,
      tf.cast(flat_shape, sparse_indices.dtype)
    )

  reconstructed_record = compression_utils.inverse_flatten_concat(x, input_record)
  return reconstructed_record


def _create_sq_fn(value_type, bits, p):
  @tff.tf_computation(value_type)
  def sq_fn(record):
    microseconds_per_second = 10 ** 6  # Timestamp returns fractional seconds.
    timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                     tf.int32)

    rand_p_seed = tf.convert_to_tensor([timestamp_microseconds * 2, 0])

    return sq_roundtrip(record,
                        rand_p_seed=rand_p_seed,
                        p=p,
                        bits=bits)

  return sq_fn
