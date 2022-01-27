from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
from tensorflow_model_optimization.python.core.internal.tensor_encoding.encoders import as_simple_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages import FlattenEncodingStage, \
  BitpackingEncodingStage
import tensorflow_federated as tff
import tensorflow as tf
from distributed_dp import compression_utils


def kashin_encoder():
  return te.core.EncoderComposer(
    te.stages.research.KashinHadamardEncodingStage(),
  ).add_parent(
    FlattenEncodingStage(),
    FlattenEncodingStage.ENCODED_VALUES_KEY
  ).make()


def sample_indices(shape, p, seed):
  mask = tf.less(tf.random.stateless_uniform(shape, seed=seed), p)
  return tf.where(mask)


def round_fn(x):
  floored_x = tf.floor(x)
  decimal_x = x - floored_x

  bernoulli = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=0, maxval=1) < decimal_x
  rounded_x = floored_x + tf.cast(bernoulli, x.dtype)

  return rounded_x


def enc_roundtrip(input_record, enc, sparsify_roundtrip, rand_p_seed, bits, p):
  """Applies compression to the record as a single concatenated vector."""
  input_vec = compression_utils.flatten_concat(input_record)

  casted_record = tf.cast(input_vec, tf.float32)

  _state = enc.initial_state()
  encode_params, decode_params = enc.get_params(_state)
  encoded_tensors, state_update_tensors, input_shapes = enc.encode(casted_record, encode_params)
  x = encoded_tensors['flattened_values']['kashin_hadamard_values']
  max_int_value = tf.cast(2 ** bits - 1, x.dtype)
  offset, x_max = tf.reduce_min(x), tf.reduce_max(x)
  scale = x_max - offset
  quantized_record = round_fn(tf.math.divide_no_nan(x - offset, scale) * max_int_value)
  x = quantized_record / max_int_value * scale + offset

  if p < 1:
    x = sparsify_roundtrip(x, rand_p_seed)

  encoded_tensors['flattened_values']['kashin_hadamard_values'] = x

  decoded_x = enc.decode(encoded_tensors, decode_params, input_shapes)

  if input_vec.dtype.is_integer:
    uncasted_input_vec = tf.cast(tf.round(decoded_x), input_vec.dtype)
  else:
    uncasted_input_vec = tf.cast(decoded_x, input_vec.dtype)

  reconstructed_record = compression_utils.inverse_flatten_concat(
    uncasted_input_vec, input_record)
  return reconstructed_record


def _create_kashin_fn(value_type, bits, p):
  enc = kashin_encoder()

  @tf.function
  def sparsify_roundtrip(x, rand_p_seed):
    sparse_indices = sample_indices(x.shape, 1 - p, rand_p_seed)
    z = tf.zeros(tf.shape(sparse_indices)[0], dtype=x.dtype)
    x = tf.tensor_scatter_nd_update(x, sparse_indices, z)
    return x * (1 / p)

  @tff.tf_computation(value_type)
  def kashin_fn(record):
    microseconds_per_second = 10 ** 6  # Timestamp returns fractional seconds.
    timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                     tf.int32)

    rand_p_seed = tf.convert_to_tensor([timestamp_microseconds * 2, 0])

    return enc_roundtrip(record,
                         enc=enc,
                         sparsify_roundtrip=sparsify_roundtrip,
                         rand_p_seed=rand_p_seed,
                         bits=bits,
                         p=p)

  return kashin_fn
