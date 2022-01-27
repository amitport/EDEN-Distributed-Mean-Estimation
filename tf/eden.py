import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.ops import math_ops

from distributed_dp import compression_utils

### half-normal centroids

half_centroids = {1: [0.7978845608028654],
                  2: [0.4527800398860679, 1.5104176087114887],
                  3: [0.24509416307340598, 0.7560052489539643,
                      1.3439092613750225, 2.151945669890335],
                  4: [0.12839501671105813, 0.38804823445328507,
                      0.6567589957631145, 0.9423402689122875,
                      1.2562309480263467, 1.6180460517130526, 2.069016730231837,
                      2.732588804065177],
                  5: [0.06588962234909321, 0.1980516892038791,
                      0.3313780514298761, 0.4666991751197207,
                      0.6049331689395434, 0.7471351317890572, 0.89456439585444,
                      1.0487823813655852, 1.2118032120324,
                      1.3863389353626248, 1.576226389073775, 1.7872312118858462,
                      2.0287259913633036, 2.3177364021261493,
                      2.69111557955431, 3.260726295605043],
                  6: [0.0334094558802581, 0.1002781217139195,
                      0.16729660990171974, 0.23456656976873475,
                      0.3021922894403614, 0.37028193328115516,
                      0.4389488009177737, 0.5083127587538033,
                      0.5785018460645791, 0.6496542452315348,
                      0.7219204720694183, 0.7954660529025513, 0.870474868055092,
                      0.9471530930156288, 1.0257343133937524,
                      1.1064859596918581, 1.1897175711327463,
                      1.2757916223519965,
                      1.3651378971823598, 1.458272959944728, 1.5558274659528346,
                      1.6585847114298427, 1.7675371481292605,
                      1.8839718992293555, 2.009604894545278, 2.146803022259123,
                      2.2989727412973995, 2.471294740528467,
                      2.6722617014102585, 2.91739146530985, 3.2404166403241677,
                      3.7440690236964755],
                  7: [0.016828143177728235, 0.05049075396896167,
                      0.08417241989671888, 0.11788596825032507,
                      0.1516442630131618, 0.18546025708680833,
                      0.21934708340331643, 0.25331807190834565,
                      0.2873868062260947, 0.32156710392315796,
                      0.355873075050329, 0.39031926330596733,
                      0.4249205523979007, 0.4596922300454219,
                      0.49465018161031576, 0.5298108436256188,
                      0.565191195643323,
                      0.600808970989236, 0.6366826613981411, 0.6728315674936343,
                      0.7092759460939766, 0.746037126679468,
                      0.7831375375631398, 0.8206007832455021, 0.858451939611374,
                      0.896717615963322, 0.9354260757626341,
                      0.9746074842160436, 1.0142940678300427, 1.054520418037026,
                      1.0953237719213182, 1.1367442623434032,
                      1.1788252655205043, 1.2216138763870124, 1.26516137869917,
                      1.309523700469555, 1.3547621051156036,
                      1.4009441065262136, 1.448144252238147, 1.4964451375010575,
                      1.5459387008934842, 1.596727786313424,
                      1.6489283062238074, 1.7026711624156725,
                      1.7581051606756466, 1.8154009933798645,
                      1.8747553268072956,
                      1.9363967204122827, 2.0005932433837565,
                      2.0676621538384503, 2.1379832427349696, 2.212016460501213,
                      2.2903268704925304, 2.3736203164211713,
                      2.4627959084523208, 2.5590234991374485, 2.663867022558051,
                      2.7794919110540777, 2.909021527386642, 3.0572161028423737,
                      3.231896182843021, 3.4473810105937095,
                      3.7348571053691555, 4.1895219330235225],
                  8: [0.008445974137017219, 0.025338726226901278,
                      0.042233889994651476, 0.05913307399220878,
                      0.07603788791797023, 0.09294994306815242,
                      0.10987089037069565, 0.12680234584461386,
                      0.1437459285205906, 0.16070326074968388,
                      0.1776760066764216, 0.19466583496246115,
                      0.21167441946986007, 0.22870343946322488,
                      0.24575458029044564, 0.2628295721769575,
                      0.2799301528634766, 0.29705806782573063,
                      0.3142150709211129, 0.3314029639954903,
                      0.34862355883476864, 0.3658786774238477,
                      0.3831701926964899, 0.40049998943716425,
                      0.4178699650069057, 0.4352820704086704,
                      0.45273827097956804, 0.4702405882876, 0.48779106011037887,
                      0.505391740756901, 0.5230447441905988, 0.5407522460590347,
                      0.558516486141511, 0.5763396823538222,
                      0.5942241184949506, 0.6121721459546814,
                      0.6301861414640443, 0.6482685527755422,
                      0.6664219019236218,
                      0.684648787627676, 0.7029517931200633, 0.7213336286470308,
                      0.7397970881081071, 0.7583450032075904,
                      0.7769802937007926, 0.7957059197645721,
                      0.8145249861674053, 0.8334407494351099,
                      0.8524564651728141,
                      0.8715754936480047, 0.8908013031010308,
                      0.9101374749919184, 0.9295877653215154,
                      0.9491559977740125,
                      0.9688461234581733, 0.9886622867721733,
                      1.0086087121824747, 1.028689768268861, 1.0489101021225093,
                      1.0692743940997251, 1.0897875553561465,
                      1.1104547388972044, 1.1312812154370708,
                      1.1522725891384287,
                      1.173434599389649, 1.1947731980672593, 1.2162947131430126,
                      1.238005717146854, 1.2599130381874064,
                      1.2820237696510286, 1.304345369166531, 1.3268857708606756,
                      1.349653145284911, 1.3726560932224416,
                      1.3959037693197867, 1.419405726021264, 1.4431719292973744,
                      1.4672129964566984, 1.4915401336751468,
                      1.5161650628244996, 1.541100284490976, 1.5663591473033147,
                      1.5919556551358922, 1.6179046397057497,
                      1.6442219553485078, 1.6709244249695359,
                      1.6980300628044107, 1.7255580190748743,
                      1.7535288357430767,
                      1.7819645728459763, 1.81088895442524, 1.8403273195729115,
                      1.870306964218662, 1.9008577747790962,
                      1.9320118435829472, 1.9638039107009146,
                      1.9962716117712092, 2.0294560760505993,
                      2.0634026367482017,
                      2.0981611002741527, 2.133785932225919, 2.170336784741086,
                      2.2078803102947337, 2.2464908293749546,
                      2.286250990303635, 2.327254033532845, 2.369604977942217,
                      2.4134218838650208, 2.458840003415269,
                      2.506014300608167, 2.5551242195294983, 2.6063787537827645,
                      2.660023038604595, 2.716347847697055,
                      2.7757011083910723, 2.838504606698991, 2.9052776685316117,
                      2.976670770545963, 3.0535115393558603,
                      3.136880130166507, 3.2282236667414654, 3.3295406612081644,
                      3.443713971315384, 3.5751595986789093,
                      3.7311414987004117, 3.9249650523739246, 4.185630113705256,
                      4.601871059539151]}


### centroids to bin boundaries

def gen_boundaries(centroids):
  return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]


half_boundaries = {i: gen_boundaries(c) for i, c in half_centroids.items()}

### add symmetric negative normal centroids
all_centroids = {i: [-j for j in reversed(c)] + c for i, c in
                 half_centroids.items()}
all_boundaries = {i: [*[-j for j in reversed(c)], 0, *c] for i, c in
                  half_boundaries.items()}


def eden_quantization(x, bits):
  centroids = all_centroids[bits]
  boundaries = all_boundaries[bits]

  # assign quantization levels
  d = tf.cast(tf.size(x), x.dtype)
  ss = tf.reduce_sum(x ** 2)
  l2 = tf.sqrt(ss)

  assignments = math_ops.bucketize(x * (d ** 0.5) / l2, boundaries)

  unscaled_centers_vec = tf.gather(centroids, assignments)

  scale = ss / tf.reduce_sum(unscaled_centers_vec * x)

  return assignments, scale


def inverse_eden_quantization(assignments, scale, bits):
  centroids = all_centroids[bits]

  unscaled_centers_vec = tf.gather(centroids, assignments)

  # restore scale
  return scale * unscaled_centers_vec


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
