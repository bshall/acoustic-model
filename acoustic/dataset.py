import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = Path(root)
        split = "train.txt" if train else "validation.txt"
        with open(self.root / split) as file:
            self.metadata = [self.root / line.strip() for line in file]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]

        mel = np.load(path.with_suffix(".npy")).T
        units = np.load(path.with_suffix(".npy"))

        length = min(mel.shape[0], units.shape[0])
        # length = 2 * codes.shape[0]
        # mel, codes = np.pad(mel[:length, :], ((1, 0), (0, 0))), codes[:length]
        # mel, codes = np.pad(mel[:length, :], ((1, 0), (0, 0))), codes
        mel, units = np.pad(mel[:length, :], ((1, 0), (0, 0))), units[:length, :]

        return torch.from_numpy(mel), torch.tensor(units)


def pad_collate(batch):
    mels, codes = zip(*batch)

    mels, codes = list(mels), list(codes)

    mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
    codes_lengths = torch.tensor([x.size(0) for x in codes])

    mels = pad_sequence(mels, batch_first=True)
    # codes = pad_sequence(codes, batch_first=True, padding_value=100)
    codes = pad_sequence(codes, batch_first=True)

    return mels, mels_lengths, codes, codes_lengths
