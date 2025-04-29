import glob
import numpy as np
import note_seq
import torch

from torch.utils.data import Dataset
from converters import groove
from constants import DATADIRS, DRUM_PITCH_CLASSES, SEQUENCE_LENGTH
from tqdm import tqdm
import json


class RhythmDataset(Dataset):
    def __init__(
        self,
        dataset_name="gmd",
        splits=[0.8, 0.1, 0.1],
        shuffle=False,
        split="train",
        humanize=False,
        tapify=True,
        _fixed_velocities=True
    ):
        try:
            files = glob.glob(f"{DATADIRS[dataset_name]}/*/*/*.mid")
        except KeyError as e:
            print(f"please check data/constants.py that a directory for {dataset_name} exists")
            raise e
        if len(files) == 0:
            print(f"no files found in {DATADIRS[dataset_name]}")
            raise FileNotFoundError

        print(len(files))
        self.data = self._create_splits(files, splits, shuffle)
        self.files = self.data[split]
        print(len(self.files))

        try:
            self.pitch_classes = DRUM_PITCH_CLASSES[dataset_name]
        except KeyError:
            print(f"check data/constants that a pitch class for {dataset_name} exists")
            raise
        self.humanize = humanize
        self.tapify = tapify
        self.fix_velocity = _fixed_velocities
        self.input_size = (32, 27)
        self.tensor_list = self.load()
        with open("/data/GrooTransformer/data/vocab.json", mode="r") as fin:
            self.vocab_set = json.load(fin)
        np.random.shuffle(self.tensor_list)

    def midi_to_tensor(self, fname: str):
        with open(fname, "rb") as f:
            sequence = note_seq.midi_io.midi_to_note_sequence(f.read())
            # quantized_sequence = note_seq.sequences_lib.quantize_note_sequence(
            #     sequence, steps_per_quarter=4
            # )
        converter = groove.GrooveConverter(
            split_bars=2,
            steps_per_quarter=4,
            quarters_per_bar=4,
            pitch_classes=self.pitch_classes,
            humanize=self.humanize,
            tapify=self.tapify,
            fixed_velocities=self.fix_velocity
        )
        tensor = converter.to_tensors(sequence) #quantized_sequence
        return tensor

    @staticmethod
    def _create_splits(files, splits, shuffle):
        if shuffle:
            idx = np.random.permutation(len(files))
            files = [files[i] for i in idx]
        idx = np.linspace(0, len(files) - 1, len(files)).astype("int")

        train_idx = idx[: int(splits[0] * len(files))]
        valid_idx = idx[int(splits[0] * len(files)): int((splits[0] + splits[1]) * len(files))]
        test_idx = idx[int((splits[0] + splits[1]) * len(files)):]

        train = [files[i] for i in train_idx]
        valid = [files[i] for i in valid_idx]
        test = [files[i] for i in test_idx]

        return {"train": train, "valid": valid, "test": test}

    def __getitem__(self, idx):
        # mask = (np.argmax(self.tensor_list[idx][0][:, :9], axis=1) > 0).astype("float")
        taps = torch.as_tensor(self.tensor_list[idx][0][:, 3]).type(torch.LongTensor).cuda()
        drums_arr = self.tensor_list[idx][1][:, :9]
        drums = []
        for i in range(drums_arr.shape[0]):
            key = "".join(drums_arr[i].astype("int").astype("str").tolist())
            drums.append(self.vocab_set[key])
        drums = torch.from_numpy(np.array(drums)).type(torch.LongTensor).cuda()
        return taps, drums


    def __len__(self):
        return self.length

    def load(self):
        tensor_list = []
        for i in tqdm(range(len(self.files))):
            try:
                tensor = self.midi_to_tensor(self.files[i])

                if len(tensor.inputs) == 0:
                    print("Problem with {}".format(self.files[i]))
                for k in range(len(tensor.inputs)):
                    inp = tensor.inputs[k]
                    out = tensor.outputs[k]
                    tensor_list.append((inp, out))
            except Exception as e:
                print("Problem with {}".format(self.files[i]))
        self.length = len(tensor_list)
        print("Length of data: {}".format(len(tensor_list)))
        return tensor_list
