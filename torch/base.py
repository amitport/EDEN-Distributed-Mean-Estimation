from abc import ABC, abstractmethod


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
