""" Torch Base Dataset Class"""
import torch
import numpy as np
from deep_math.constants import BOS, EOS


class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset Class. All dataset will have this class as type

    Args:
        torch (Module): torch Dataset Module
    """

    def __init__(self, data, targets, transform=None, target_transform=None):
        super().__init__()
        if len(data) != len(targets):
            print("Length of Targets must match with length of Data")

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key.

        Args:
            idx (int): index of the data point
        """
        data_point = self.data[idx]
        label = self.targets[idx]

        # If transform is available then return transformed dataPoint and label
        if self.transform is not None:
            data_point = self.transform(data_point)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return {
            "q": data_point,
            "q_enc": np_encode_string(data_point),
            "a": label,
            "a_enc": np_encode_string(label),
        }

    def __len__(self):
        """expected to return the size of the dataset
        """
        return len(self.data)


def np_encode_string(s, char0=ord(" ")):
    """converts a string into a numpy array of bytes
    (char0 - 1) is subtracted from all bytes values (0 is used for PAD)
    string is pre-pended with BOS and post-pended with EOS"""
    chars = np.array(list(s), dtype="S1").view(np.uint8)
    # normalize to 1 - 96, 0 being PAD
    chars = chars - char0 + 1

    chars = np.insert(chars, 0, BOS)
    chars = np.insert(chars, len(chars), EOS)
    return chars


def np_decode_string(chars, char0=ord(" ")):
    """converts a numpy array of bytes into a UTF-8 string
    (char0 - 1) is added to all bytes values (0 is used for PAD)
    BOS/EOS are removed before utf-8 decoding"""
    chars = chars.astype(np.uint8)
    chars = chars + char0 - 1
    chars = chars[:-1]
    chars = chars.tobytes()
    s = chars.decode("UTF-8")
    return s