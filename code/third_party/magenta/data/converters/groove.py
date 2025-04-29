import collections
import copy
import note_seq
import numpy as np

from .base import BaseNoteSequenceConverter, ConverterTensors


def np_onehot(indices, depth, dtype=np.bool):
    """Converts 1D array of indices to a one-hot 2D array with given depth."""
    onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
    onehot_seq[np.arange(len(indices)), indices] = 1.0
    return onehot_seq


class GrooveConverter(BaseNoteSequenceConverter):
    """Converts to and from hit/velocity/offset representations.
  In this setting, we represent drum sequences and performances
  as triples of (hit, velocity, offset). Each timestep refers to a fixed beat
  on a grid, which is by default spaced at 16th notes.  Drum hits that don't
  fall exactly on beat are represented through the offset value, which refers
  to the relative distance from the nearest quantized step.
  Hits are binary [0, 1].
  Velocities are continuous values in [0, 1].
  Offsets are continuous values in [-0.5, 0.5], rescaled to [-1, 1] for tensors.
  Each timestep contains this representation for each of a fixed list of
  drum categories, which by default is the list of 9 categories defined in
  drums_encoder_decoder.py.  With the default categories, the input and output
  at a single timestep is of length 9x3 = 27. So a single measure of drums
  at a 16th note grid is a matrix of shape (16, 27).
  Attributes:
    split_bars: Optional size of window to slide over full converted tensor.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups Roland V-Drum pitches into 9 different classes.
    inference_pitch_classes: Pitch classes to use during inference. By default,
      uses same as `pitch_classes`.
    humanize: If True, flatten all input velocities and microtiming. The model
      then learns to map from a flattened input to the original sequence.
    tapify: If True, squash all drums at each timestep to the open hi-hat
      channel.
    add_instruments: A list of strings matching drums in DRUM_LIST.
      These drums are removed from the inputs but not the outputs.
    num_velocity_bins: The number of bins to use for representing velocity as
      one-hots.  If not defined, the converter will use continuous values.
    num_offset_bins: The number of bins to use for representing timing offsets
      as one-hots.  If not defined, the converter will use continuous values.
    split_instruments: Whether to produce outputs for each drum at a given
      timestep across multiple steps of the model output. With 9 drums, this
      makes the sequence 9 times as long. A one-hot control sequence is also
      created to identify which instrument is to be output at each step.
    hop_size: Number of steps to slide window.
    hits_as_controls: If True, pass in hits with the conditioning controls
      to force model to learn velocities and offsets.
    fixed_velocities: If True, flatten all input velocities.
    max_note_dropout_probability: If a value is provided, randomly drop out
      notes from the input sequences but not the output sequences.  On a per
      sequence basis, a dropout probability will be chosen uniformly between 0
      and this value such that some sequences will have fewer notes dropped
      out and some will have have more.  On a per note basis, lower velocity
      notes will be dropped out more often.
  """

    def __init__(
        self,
        split_bars=None,
        steps_per_quarter=4,
        quarters_per_bar=4,
        max_tensors_per_notesequence=8,
        pitch_classes=None,
        inference_pitch_classes=None,
        humanize=False,
        tapify=False,
        add_instruments=None,
        num_velocity_bins=None,
        num_offset_bins=None,
        split_instruments=False,
        hop_size=None,
        hits_as_controls=False,
        fixed_velocities=False,
        max_note_dropout_probability=None,
    ):

        self._split_bars = split_bars
        self._steps_per_quarter =  steps_per_quarter
        self._steps_per_bar = steps_per_quarter * quarters_per_bar

        self._humanize = humanize
        self._tapify = tapify
        self._add_instruments = add_instruments
        self._fixed_velocities = fixed_velocities

        self._num_velocity_bins = num_velocity_bins
        self._num_offset_bins = num_offset_bins
        self._categorical_outputs = num_velocity_bins and num_offset_bins

        self._split_instruments = split_instruments

        self._hop_size = hop_size
        self._hits_as_controls = hits_as_controls

        def _classes_to_map(classes):
            class_map = {}
            for cls, pitches in enumerate(classes):
                for pitch in pitches:
                    class_map[pitch] = cls
            return class_map

        self._pitch_classes = pitch_classes
        self._pitch_class_map = _classes_to_map(self._pitch_classes)
        self._infer_pitch_classes = inference_pitch_classes or self._pitch_classes
        self._infer_pitch_class_map = _classes_to_map(self._infer_pitch_classes)
        if len(self._pitch_classes) != len(self._infer_pitch_classes):
            raise ValueError(
                "Training and inference must have the same number of pitch classes. "
                "Got: %d vs %d."
                % (len(self._pitch_classes), len(self._infer_pitch_classes))
            )
        self._num_drums = len(self._pitch_classes)

        if bool(num_velocity_bins) ^ bool(num_offset_bins):
            raise ValueError(
                "Cannot define only one of num_velocity_vins and num_offset_bins."
            )

        if split_bars is None and hop_size is not None:
            raise ValueError("Cannot set hop_size without setting split_bars")

        drums_per_output = 1 if self._split_instruments else self._num_drums
        # Each drum hit is represented by 3 numbers - on/off, velocity, and offset
        if self._categorical_outputs:
            output_depth = drums_per_output * (1 + num_velocity_bins + num_offset_bins)
        else:
            output_depth = drums_per_output * 3

        control_depth = 0
        # Set up controls for passing hits as side information.
        if self._hits_as_controls:
            if self._split_instruments:
                control_depth += 1
            else:
                control_depth += self._num_drums
        # Set up controls for cycling through instrument outputs.
        if self._split_instruments:
            control_depth += self._num_drums

        self._max_note_dropout_probability = max_note_dropout_probability
        self._note_dropout = max_note_dropout_probability is not None

        super(GrooveConverter, self).__init__(
            input_depth=output_depth,
            input_dtype=np.float32,
            output_depth=output_depth,
            output_dtype=np.float32,
            control_depth=control_depth,
            control_dtype=np.bool,
            end_token=False,
            presplit_on_time_changes=False,
            max_tensors_per_notesequence=max_tensors_per_notesequence,
        )

    @property
    def pitch_classes(self):
        if self.is_inferring:
            return self._infer_pitch_classes
        return self._pitch_classes

    @property
    def pitch_class_map(self):  # pylint: disable=g-missing-from-attributes
        if self.is_inferring:
            return self._infer_pitch_class_map
        return self._pitch_class_map

    def _get_feature(self, note, feature, step_length=None):
        """Compute numeric value of hit/velocity/offset for a note.
    For now, only allow one note per instrument per quantization time step.
    This means at 16th note resolution we can't represent some drumrolls etc.
    We just take the note with the highest velocity if there are multiple notes.
    Args:
      note: A Note object from a NoteSequence.
      feature: A string, either 'hit', 'velocity', or 'offset'.
      step_length: Time duration in seconds of a quantized step. This only needs
        to be defined when the feature is 'offset'.
    Raises:
      ValueError: Any feature other than 'hit', 'velocity', or 'offset'.
    Returns:
      The numeric value of the feature for the note.
    """

        def _get_offset(note, step_length):
            true_onset = note.start_time
            quantized_onset = step_length * note.quantized_start_step
            diff = quantized_onset - true_onset
            return diff / step_length

        if feature == "hit":
            if note:
                return 1.0
            else:
                return 0.0

        elif feature == "velocity":
            if note:
                return (
                    note.velocity / 127.0
                )  # Switch from [0, 127] to [0, 1] for tensors.
            else:
                return 0.0  # Default velocity if there's no note is 0

        elif feature == "offset":
            if note:
                offset = _get_offset(note, step_length)
                return offset * 2  # Switch from [-0.5, 0.5] to [-1, 1] for tensors.
            else:
                return 0.0  # Default offset if there's no note is 0

        else:
            raise ValueError("Unlisted feature: " + feature)

    def to_tensors(self, note_sequence):
        def _get_steps_hash(note_sequence):
            """Partitions all Notes in a NoteSequence by quantization step and drum.
      Creates a hash with each hash bucket containing a dictionary
      of all the notes at one time step in the sequence grouped by drum/class.
      If there are no hits at a given time step, the hash value will be {}.
      Args:
        note_sequence: The NoteSequence object
      Returns:
        The fully constructed hash
      Raises:
        ValueError: If the sequence is not quantized
      """
            if not note_seq.sequences_lib.is_quantized_sequence(note_sequence):
                raise ValueError("NoteSequence must be quantized")

            h = collections.defaultdict(lambda: collections.defaultdict(list))

            for note in note_sequence.notes:
                step = int(note.quantized_start_step)
                drum = self.pitch_class_map[note.pitch]
                h[step][drum].append(note)

            return h

        def _remove_drums_from_tensors(to_remove, tensors):
            """Drop hits in drum_list and set velocities and offsets to 0."""
            for t in tensors:
                t[:, to_remove] = 0.0
            return tensors

        def _convert_vector_to_categorical(vectors, min_value, max_value, num_bins):
            # Avoid edge case errors by adding a small amount to max_value
            bins = np.linspace(min_value, max_value + 0.0001, num_bins)
            return np.array(
                [
                    np.concatenate(
                        np_onehot(
                            np.digitize(v, bins, right=True), num_bins, dtype=np.int32
                        )
                    )
                    for v in vectors
                ]
            )

        def _extract_windows(tensor, window_size, hop_size):
            """Slide a window across the first dimension of a 2D tensor."""
            if len(tensor) < window_size:
                return [np.concatenate([tensor, np.zeros((window_size - len(tensor), tensor.shape[1]))], axis=0)]
            return [
                tensor[i: i + window_size, :]
                for i in range(0, len(tensor) - window_size + 1, hop_size)
            ]

        try:
            quantized_sequence = note_seq.sequences_lib.quantize_note_sequence(
                note_sequence, self._steps_per_quarter
            )
            # print(quantized_sequence)
            if (
                note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence)
                != self._steps_per_bar
            ):
                return ConverterTensors()
            if not quantized_sequence.time_signatures:
                quantized_sequence.time_signatures.add(numerator=4, denominator=4)
        except (
            note_seq.BadTimeSignatureError,
            note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError,
            note_seq.MultipleTimeSignatureError,
            note_seq.MultipleTempoError,
        ):
            return ConverterTensors()

        beat_length = 60.0 / quantized_sequence.tempos[0].qpm

        step_length = beat_length / (
            quantized_sequence.quantization_info.steps_per_quarter
        )
        # print("Quarter per minute: {}, Beat length: {}, Step length: {}".format(quantized_sequence.tempos[0].qpm,
        #                                                                         beat_length,
        #                                                                         step_length))
        steps_hash = _get_steps_hash(quantized_sequence)
        # print(steps_hash)
        if not quantized_sequence.notes:
            return ConverterTensors()

        max_start_step = np.max(
            [note.quantized_start_step for note in quantized_sequence.notes]
        )

        # Round up so we pad to the end of the bar.
        total_bars = int(np.ceil((max_start_step + 1) / self._steps_per_bar))
        max_step = self._steps_per_bar * total_bars
        # print("Max step: {}, total bars: {}".format(max_step, total_bars))
        # Each of these stores a (total_beats, num_drums) matrix.
        hit_vectors = np.zeros((max_step, self._num_drums))
        velocity_vectors = np.zeros((max_step, self._num_drums))
        offset_vectors = np.zeros((max_step, self._num_drums))

        # Loop through timesteps.
        for step in range(max_step):
            notes = steps_hash[step]

            # Loop through each drum instrument.
            for drum in range(self._num_drums):
                drum_notes = notes[drum]
                if len(drum_notes) > 1:
                    note = max(drum_notes, key=lambda n: n.velocity)
                elif len(drum_notes) == 1:
                    note = drum_notes[0]
                else:
                    note = None

                hit_vectors[step, drum] = self._get_feature(note, "hit")
                velocity_vectors[step, drum] = self._get_feature(note, "velocity")
                offset_vectors[step, drum] = self._get_feature(
                    note, "offset", step_length
                )

        # These are the input tensors for the encoder.
        in_hits = copy.deepcopy(hit_vectors)
        in_velocities = copy.deepcopy(velocity_vectors)
        in_offsets = copy.deepcopy(offset_vectors)

        # print(in_hits)

        if self._note_dropout:
            # Choose a uniform dropout probability for notes per sequence.
            note_dropout_probability = np.random.uniform(
                0.0, self._max_note_dropout_probability
            )
            # Drop out lower velocity notes with higher probability.
            velocity_dropout_weights = np.maximum(0.2, (1 - in_velocities))
            note_dropout_keep_mask = 1 - np.random.binomial(
                1, velocity_dropout_weights * note_dropout_probability
            )
            in_hits *= note_dropout_keep_mask
            in_velocities *= note_dropout_keep_mask
            in_offsets *= note_dropout_keep_mask

        if self._tapify:
            argmaxes = np.argmax(in_velocities, axis=1)
            in_hits[:] = 0
            in_velocities[:] = 0
            in_offsets[:] = 0
            in_hits[:, 3] = hit_vectors[np.arange(max_step), argmaxes]
            in_velocities[:, 3] = velocity_vectors[np.arange(max_step), argmaxes]
            in_offsets[:, 3] = offset_vectors[np.arange(max_step), argmaxes]

        if self._humanize:
            in_velocities[:] = 0
            in_offsets[:] = 0

        if self._fixed_velocities:
            in_velocities[:] = 0

        # If learning to add drums, remove the specified drums from the inputs.
        if self._add_instruments:
            in_hits, in_velocities, in_offsets = _remove_drums_from_tensors(
                self._add_instruments, [in_hits, in_velocities, in_offsets]
            )

        if self._categorical_outputs:
            # Convert continuous velocity and offset to one hots.
            velocity_vectors = _convert_vector_to_categorical(
                velocity_vectors, 0.0, 1.0, self._num_velocity_bins
            )
            in_velocities = _convert_vector_to_categorical(
                in_velocities, 0.0, 1.0, self._num_velocity_bins
            )

            offset_vectors = _convert_vector_to_categorical(
                offset_vectors, -1.0, 1.0, self._num_offset_bins
            )
            in_offsets = _convert_vector_to_categorical(
                in_offsets, -1.0, 1.0, self._num_offset_bins
            )

        if self._split_instruments:
            # Split the outputs for each drum into separate steps.
            total_length = max_step * self._num_drums
            hit_vectors = hit_vectors.reshape([total_length, -1])
            velocity_vectors = velocity_vectors.reshape([total_length, -1])
            offset_vectors = offset_vectors.reshape([total_length, -1])
            in_hits = in_hits.reshape([total_length, -1])
            in_velocities = in_velocities.reshape([total_length, -1])
            in_offsets = in_offsets.reshape([total_length, -1])
        else:
            total_length = max_step

        # Now concatenate all 3 vectors into 1, eg (16, 27).
        seqs = np.concatenate([hit_vectors, velocity_vectors, offset_vectors], axis=1)

        input_seqs = np.concatenate([in_hits, in_velocities, in_offsets], axis=1)
        # print(input_seqs)

        # Controls section.
        controls = []
        if self._hits_as_controls:
            controls.append(hit_vectors.astype(np.bool))
        if self._split_instruments:
            # Cycle through instrument numbers.
            controls.append(
                np.tile(
                    np_onehot(np.arange(self._num_drums), self._num_drums, np.bool),
                    (max_step, 1),
                )
            )
        controls = np.concatenate(controls, axis=-1) if controls else None

        if self._split_bars:
            window_size = self._steps_per_bar * self._split_bars
            hop_size = self._hop_size or window_size
            if self._split_instruments:
                window_size *= self._num_drums
                hop_size *= self._num_drums
            seqs = _extract_windows(seqs, window_size, hop_size)
            input_seqs = _extract_windows(input_seqs, window_size, hop_size)
            # print("input seqs", len(input_seqs))
            if controls is not None:
                controls = _extract_windows(controls, window_size, hop_size)
        else:
            # Output shape will look like (1, 64, output_depth).
            seqs = [seqs]
            input_seqs = [input_seqs]
            if controls is not None:
                controls = [controls]
        # print("FFFFFFFFFF", len(input_seqs))
        # for i in range(len(input_seqs)):
        #     print(input_seqs[i].shape)
        return ConverterTensors(inputs=input_seqs, outputs=seqs, controls=controls)

    def from_tensors(self, samples, controls=None, qpm=120):
        def _zero_one_to_velocity(val):
            output = int(np.round(val * 127))
            return np.clip(output, 0, 127)

        def _minus_1_1_to_offset(val):
            output = val / 2
            return np.clip(output, -0.5, 0.5)

        def _one_hot_to_velocity(v):
            return int((np.argmax(v) / len(v)) * 127)

        def _one_hot_to_offset(v):
            return (np.argmax(v) / len(v)) - 0.5

        output_sequences = []

        for sample in samples:
            n_timesteps = sample.shape[0] // (
                self._num_drums if self._categorical_outputs else 1
            )

            note_sequence = note_seq.NoteSequence()
            note_sequence.tempos.add(qpm=qpm)
            beat_length = 60.0 / note_sequence.tempos[0].qpm
            step_length = beat_length / self._steps_per_quarter

            # Each timestep should be a (1, output_depth) vector
            # representing n hits, n velocities, and n offsets in order.

            for i in range(n_timesteps):
                if self._categorical_outputs:
                    # Split out the categories from the flat output.
                    if self._split_instruments:
                        (
                            hits,
                            velocities,
                            offsets,
                        ) = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                            sample[i * self._num_drums: (i + 1) * self._num_drums],
                            [1, self._num_velocity_bins + 1],
                            axis=1,
                        )
                    else:
                        (
                            hits,
                            velocities,
                            offsets,
                        ) = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                            sample[i],
                            [
                                self._num_drums,
                                self._num_drums * (self._num_velocity_bins + 1),
                            ],
                        )
                        # Split out the instruments.
                        velocities = np.split(velocities, self._num_drums)
                        offsets = np.split(offsets, self._num_drums)
                else:
                    if self._split_instruments:
                        hits, velocities, offsets = sample[
                            i * self._num_drums: (i + 1) * self._num_drums
                        ].T
                    else:
                        (
                            hits,
                            velocities,
                            offsets,
                        ) = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                            sample[i], 3
                        )

                # Loop through the drum instruments: kick, snare, etc.
                for j in range(len(hits)):
                    # Create a new note
                    if hits[j] > 0.5:
                        note = note_sequence.notes.add()
                        note.instrument = 9  # All drums are instrument 9
                        note.is_drum = True
                        pitch = self.pitch_classes[j][0]
                        note.pitch = pitch
                        if self._categorical_outputs:
                            note.velocity = _one_hot_to_velocity(velocities[j])
                            offset = _one_hot_to_offset(offsets[j])
                        else:
                            note.velocity = _zero_one_to_velocity(velocities[j])
                            offset = _minus_1_1_to_offset(offsets[j])
                        note.start_time = (i - offset) * step_length
                        note.end_time = note.start_time + step_length
                        # print(note.end_time)

            output_sequences.append(note_sequence)

        return output_sequences