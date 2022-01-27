from functools import partial

import tensorflow_federated as tff
from absl import logging

from eden import _create_eden_fn
from hadamard import _create_hadamard_fn
from kashin import _create_kashin_fn
from sq import _create_sq_fn

SUPPORTED_COMPRESSORS = ['noop', 'eden', 'hadamard', 'kashin', 'sq']


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
