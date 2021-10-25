from genericpath import exists
from pathlib import Path
import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from deep_math.constants import PAD, MAX_ANSWER_SZ, MAX_QUESTION_SZ, DATA_ID, MINI, SIMPLE_LSTM
from deep_math.data.base_dataset import BaseDataset
from deep_math.data.util import download_file_from_google_drive, load_and_print_info
import zipfile
from tqdm import tqdm
import shutil
import pytorch_lightning as pl
from deep_math.util import lstm_batch_collate_fn  # For testing purpose

# Define file paths
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
PROCESSED_DATA_DIRNAME = DATA_DIR / "mathematics_dataset-v1.0"
MINI_DATA_DIRNAME = DATA_DIR / "mini_mathematics_dataset-v1.0"


class MathDataModule(pl.LightningDataModule):

    def __init__(self, collate_fn, dataset_type=MINI):
        super().__init__()
        self.collate_fn = collate_fn
        self.batch_size = 8
        self.num_workers = 4  # Count of subprocesses to use for data loading
        self.max_elements = 5
        self.dataset_type = dataset_type

        self.train_dirs = {
            "train-easy": PROCESSED_DATA_DIRNAME / "train-easy",
            "train-medium": PROCESSED_DATA_DIRNAME / "train-medium",
            "train-hard": PROCESSED_DATA_DIRNAME / "train-hard",
        }
        self.train_dirs_mini = {
            "train-easy": MINI_DATA_DIRNAME / "train-easy",
            "train-medium": MINI_DATA_DIRNAME / "train-medium",
            "train-hard": MINI_DATA_DIRNAME / "train-hard",
        }

        self.val_dirs = {
            "interpolate": PROCESSED_DATA_DIRNAME / "interpolate",
        }
        self.val_dirs_mini = {
            "interpolate": MINI_DATA_DIRNAME / "interpolate",
        }

        self.test_dirs = {"extrapolate": PROCESSED_DATA_DIRNAME / "extrapolate"}
        self.test_dirs_mini = {"extrapolate": MINI_DATA_DIRNAME / "extrapolate"}

        self.split_dirs = {
            'train': self.train_dirs,
            'val': self.val_dirs,
            'test': self.test_dirs
        }
        self.split_dirs_mini = {
            'train': self.train_dirs_mini,
            'val': self.val_dirs_mini,
            'test': self.test_dirs_mini
        }

        self.math_dataset = {'data_train': [], 'data_val': [], 'data_test': []}

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists(PROCESSED_DATA_DIRNAME):
            _download_and_process_math_dataset()
        if ((self.dataset_type == MINI) and
            (not os.path.exists(MINI_DATA_DIRNAME))):
            _create_mini_dataset(self.split_dirs)

    def setup(self, stage=None):
        data_dirs = {}
        if (self.dataset_type == MINI):
            data_dirs = self.split_dirs_mini
        else:
            data_dirs = self.split_dirs
        splits = list(data_dirs.keys())
        for split in splits:
            print(
                f"Loading {split} data with max_elements: {self.max_elements}")

            dirs = data_dirs[split]
            file_paths = [
                ff for key, dir in dirs.items()
                for ff in glob.glob(str(dir) + "/**/*.txt", recursive=True)
            ]
            print(f"File count: {len(file_paths)}")
            if len(file_paths) == 0:
                raise ValueError(
                    f"No files found. Are you sure { MINI_DATA_DIRNAME if self.dataset_type == MINI else PROCESSED_DATA_DIRNAME} is the correct root directory?"
                )
            data_index = 0
            data = {"questions": [], "answers": [], "original_index": []}
            for questions, answers in map(self._get_questions_answers_from_file,
                                          file_paths):
                data["questions"].extend(questions)
                data["answers"].extend(answers)
                data["original_index"] = data_index
                data_index += 1

            self.math_dataset['data_' + split] = BaseDataset(
                data["questions"], data["answers"])

    def _get_questions_answers_from_file(self, filepath):
        return get_questions_answers_from_file(filepath, self.max_elements)

    def train_dataloader(self):
        return DataLoader(
            self.math_dataset['data_train'],
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.math_dataset['data_val'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.math_dataset['data_test'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


def get_questions_answers_from_file(filepath, max_elements=None):
    count = 0
    with open(filepath) as datafile:
        questions = []
        answers = []
        for line in datafile:
            line = line.rstrip("\n")
            if max_elements is not None and count == (2 * max_elements):
                return questions, answers
            if count % 2 == 0:
                questions.append(line)
            else:
                answers.append(line)
            count += 1
        return questions, answers


def _download_and_process_math_dataset():
    _download_raw_dataset()
    _process_raw_dataset()


def _download_raw_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = DATA_DIR / 'mathematics_dataset-v1.0.zip'
    if filename.exists():
        return filename
    print(f"Downloading raw dataset to {filename}...")
    download_file_from_google_drive(DATA_ID, filename)
    return filename


def _process_raw_dataset():
    filename = DATA_DIR / 'mathematics_dataset-v1.0.zip'
    print(f"Unzipping {filename.name}...")

    with zipfile.ZipFile(filename, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(),
                         total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=DATA_DIR)


def _create_mini_dataset(dirs):
    print("Creating MINI DATASET")
    dirname = DATA_DIR / 'mini_mathematics_dataset-v1.0'
    if dirname.exists():
        return dirname
    dirname.mkdir(parents=True, exist_ok=True)

    math_module = 'algebra__linear_1d_composed'
    math_module_safe = 'algebra__polynomial_roots_big'

    splits = list(dirs.keys())
    for split in splits:
        for split_type in dirs[split].keys():
            split_dirname = dirname / split_type
            split_dirname.mkdir(parents=True, exist_ok=True)
            file_paths = list(dirs[split][split_type].glob(
                '{}*.txt'.format(math_module)))
            file_paths = list(dirs[split][split_type].glob('{}*.txt'.format(
                math_module_safe))) if file_paths == [] else file_paths

            print(f" {split_type} File count: {len(file_paths)}")
            for file_path in file_paths:
                dest_filename = file_path.name
                dest_filepath = split_dirname / dest_filename
                shutil.copy(file_path, dest_filepath)
    return dirname


if __name__ == "__main__":
    load_and_print_info(MathDataModule,
                        collate_fn=lstm_batch_collate_fn,
                        dataset_type=MINI)
