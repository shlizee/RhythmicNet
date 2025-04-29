import abc
import collections
import numpy as np
import note_seq


class ConverterTensors(
    collections.namedtuple(
        "ConverterTensors", ["inputs", "outputs", "controls", "lengths"]
    )
):
    """Tuple of tensors output by `to_tensors` method in converters.
  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

    def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        if lengths is None:
            lengths = [len(i) for i in inputs]
        if not controls:
            controls = [np.zeros([l, 0]) for l in lengths]
        return super(ConverterTensors, cls).__new__(
            cls, inputs, outputs, controls, lengths
        )


class BaseNoteSequenceConverter(object):
    """Base class for data converters between items and tensors.
  Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
  """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        input_depth,
        input_dtype,
        output_depth,
        output_dtype,
        control_depth=0,
        control_dtype=np.bool,
        end_token=None,
        max_tensors_per_notesequence=None,
        length_shape=(),
        presplit_on_time_changes=True,
    ):
        """Initializes BaseNoteSequenceConverter.
    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_notesequence: The maximum number of outputs to return for
          each input.
      length_shape: Shape of length returned by `to_tensor`.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
    """
        self._input_depth = input_depth
        self._input_dtype = input_dtype
        self._output_depth = output_depth
        self._output_dtype = output_dtype
        self._control_depth = control_depth
        self._control_dtype = control_dtype
        self._end_token = end_token
        self._max_tensors_per_input = max_tensors_per_notesequence
        self._str_to_item_fn = note_seq.NoteSequence.FromString
        self._mode = None
        self._length_shape = length_shape
        self._presplit_on_time_changes = presplit_on_time_changes

    def set_mode(self, mode):
        if mode not in ["train", "eval", "infer"]:
            raise ValueError("Invalid mode: %s" % mode)
        self._mode = mode

    @property
    def is_training(self):
        return self._mode == "train"

    @property
    def is_inferring(self):
        return self._mode == "infer"

    @property
    def str_to_item_fn(self):
        return self._str_to_item_fn

    @property
    def max_tensors_per_notesequence(self):
        return self._max_tensors_per_input

    @max_tensors_per_notesequence.setter
    def max_tensors_per_notesequence(self, value):
        self._max_tensors_per_input = value

    @property
    def end_token(self):
        """End token, or None."""
        return self._end_token

    @property
    def input_depth(self):
        """Dimension of inputs (to encoder) at each timestep of the sequence."""
        return self._input_depth

    @property
    def input_dtype(self):
        """DType of inputs (to encoder)."""
        return self._input_dtype

    @property
    def output_depth(self):
        """Dimension of outputs (from decoder) at each timestep of the sequence."""
        return self._output_depth

    @property
    def output_dtype(self):
        """DType of outputs (from decoder)."""
        return self._output_dtype

    @property
    def control_depth(self):
        """Dimension of control inputs at each timestep of the sequence."""
        return self._control_depth

    @property
    def control_dtype(self):
        """DType of control inputs."""
        return self._control_dtype

    @property
    def length_shape(self):
        """Shape of length returned by `to_tensor`."""
        return self._length_shape

    @abc.abstractmethod
    def to_tensors(self, item):
        """Python method that converts `item` into list of `ConverterTensors`."""
        pass

    @abc.abstractmethod
    def from_tensors(self, samples, controls=None):
        """Python method that decodes model samples into list of items."""
        pass
