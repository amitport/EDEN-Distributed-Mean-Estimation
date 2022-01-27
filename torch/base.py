from abc import ABC, abstractmethod
from typing import Sequence


class Transform(ABC):
  @abstractmethod
  def forward(self, x):
    """
    :param x: An object to transform
    :return: (tx, context) Transformed x and additional context, will be used as a input for 'backward'
    """

  # noinspection PyMethodMayBeStatic
  def backward(self, tx, context):
    """
    Inverse or a similar operation designed to act on outputs of the stage
    Defaults to no-op
    :param tx: A transformed value returned from the `forward` function
                parts of it may have passed through other transforms
    :param context: additional context needed for the backward step
    :return: An approximation of the original transformed object
             and an object containing reported info (measurements, stats, etc.,)
    """
    return tx, None

  def roundtrip(self, x):
    return self.backward(*self.forward(x))


class FunctionalTransform(Transform):
  def __init__(self, forward, backward):
    self._forward = forward
    self._backward = backward

  def forward(self, x):
    return self._forward(x)

  def backward(self, tx, context):
    return self._backward(tx, context)


class CompressionPipeline(FunctionalTransform):
  """ Onion transform """

  def __init__(self,
               transforms: Sequence[Transform]):
    """
    :param transforms: these will run in sequence during forward
                       while their backward functions are executed in reverse order during backward

                       we assume that each forward returns a tuple (x, context)
    """

    def _forward(x):
      context = []
      for t in transforms:
        x, step_context = t.forward(x)
        context.append(step_context)
      return x, context

    def _backward(tx, context):
      report = []
      for t, c in reversed(list(zip(transforms, context))):
        tx, step_report = t.backward(tx, c)
        report.insert(0, step_report)
      return tx, report

    super().__init__(_forward, _backward)


class Flatten(Transform):
  def forward(self, x):
    return x.view(-1), x.shape

  def backward(self, tx, original_shape):
    return tx.view(original_shape), None


flatten = Flatten()
