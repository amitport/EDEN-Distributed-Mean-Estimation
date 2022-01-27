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


def hadamard_roundtrip(input_record,
                       hadamard_seed: tf.Tensor,
                       rand_p_seed: tf.Tensor,
                       p=0.1,
                       bits=1):
  """Applies compression to the record as a single concatenated vector."""
  input_vec = compression_utils.flatten_concat(input_record)

  casted_record = tf.cast(input_vec, tf.float32)

  if p < 1:
    sparse_indices = sample_indices(casted_record.shape, p, rand_p_seed)
    sparse_record = tf.reshape(
      tf.gather(casted_record, sparse_indices) * (1 / p),
      [-1])
  else:
    sparse_record = casted_record

  rotated_record = compression_utils.randomized_hadamard_transform(
    sparse_record, seed_pair=hadamard_seed)

  max_int_value = tf.cast(2 ** bits - 1, rotated_record.dtype)
  offset, x_max = tf.reduce_min(rotated_record), tf.reduce_max(rotated_record)
  scale = x_max - offset
  quantized_record = round_fn(tf.math.divide_no_nan(rotated_record - offset, scale) * max_int_value)
  dequantized_record = quantized_record / max_int_value * scale + offset

  unrotated_record = compression_utils.inverse_randomized_hadamard_transform(
    dequantized_record,
    original_dim=tf.size(sparse_record),
    seed_pair=hadamard_seed)

  if p < 1:
    desparse_record = tf.scatter_nd(
      sparse_indices,
      unrotated_record,
      tf.cast(casted_record.shape, sparse_indices.dtype)
    )
  else:
    desparse_record = unrotated_record

  if input_vec.dtype.is_integer:
    uncasted_input_vec = tf.cast(tf.round(desparse_record), input_vec.dtype)
  else:
    uncasted_input_vec = tf.cast(desparse_record, input_vec.dtype)

  reconstructed_record = compression_utils.inverse_flatten_concat(
    uncasted_input_vec, input_record)
  return reconstructed_record


def _create_hadamard_fn(value_type, bits, p):
  @tff.tf_computation(value_type)
  def hadamard_fn(record):
    microseconds_per_second = 10 ** 6  # Timestamp returns fractional seconds.
    timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                     tf.int32)
    hadamard_seed = tf.convert_to_tensor([timestamp_microseconds, 0])

    rand_p_seed = tf.convert_to_tensor([timestamp_microseconds * 2, 0])

    return hadamard_roundtrip(record,
                              hadamard_seed=hadamard_seed,
                              rand_p_seed=rand_p_seed,
                              p=p,
                              bits=bits)

  return hadamard_fn
