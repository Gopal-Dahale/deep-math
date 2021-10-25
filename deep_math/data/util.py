from tqdm import tqdm
import requests
import numpy as np
from deep_math.constants import PAD, MAX_ANSWER_SZ, MAX_QUESTION_SZ
import torch


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            progress_bar.update(len(chunk))
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    progress_bar.close()


def question_answer_to_position_batch_collate_fn(qas):
    """ Gather + Pad the question/answer to the max seq length in batch """

    max_q_len = max(len(qa["q_enc"]) for qa in qas)
    max_a_len = max(len(qa["a_enc"]) for qa in qas)

    batch_qs = []
    batch_as = []
    for qa in qas:
        batch_qs.append(
            np.pad(
                qa["q_enc"],
                (0, max_q_len - len(qa["q_enc"])),
                mode="constant",
                constant_values=PAD,
            ))
        batch_as.append(
            np.pad(
                qa["a_enc"],
                (0, max_a_len - len(qa["a_enc"])),
                mode="constant",
                constant_values=PAD,
            ))

    batch_qs_pos = np.array(
        [[pos_i + 1 if w_i != PAD else 0
          for pos_i, w_i in enumerate(q)]
         for q in batch_qs])

    batch_as_pos = np.array(
        [[pos_i + 1 if w_i != PAD else 0
          for pos_i, w_i in enumerate(a)]
         for a in batch_as])

    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    batch_as = torch.LongTensor(batch_as)
    batch_as_pos = torch.LongTensor(batch_as_pos)

    return batch_qs, batch_qs_pos, batch_as, batch_as_pos


def lstm_batch_collate_fn(qas):
    """ Gather + Pad the question/answer to the max seq length in dataset """

    max_q_len = MAX_QUESTION_SZ
    max_a_len = MAX_ANSWER_SZ

    batch_qs = []
    batch_as = []
    for qa in qas:
        batch_qs.append(
            np.pad(
                qa["q_enc"],
                (0, max_q_len - len(qa["q_enc"])),
                mode="constant",
                constant_values=PAD,
            ))
        batch_as.append(
            np.pad(
                qa["a_enc"],
                (0, max_a_len - len(qa["a_enc"])),
                mode="constant",
                constant_values=PAD,
            ))

    batch_qs_pos = np.array(
        [[pos_i + 1 if w_i != PAD else 0
          for pos_i, w_i in enumerate(q)]
         for q in batch_qs])

    batch_as_pos = np.array(
        [[pos_i + 1 if w_i != PAD else 0
          for pos_i, w_i in enumerate(a)]
         for a in batch_as])

    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    batch_as = torch.LongTensor(batch_as)
    batch_as_pos = torch.LongTensor(batch_as_pos)

    return batch_qs, batch_qs_pos, batch_as, batch_as_pos


def load_and_print_info(data_module, **kwargs):
    collate_func = kwargs['collate_fn']
    dataset_type = kwargs['dataset_type']
    if collate_func is not None:
        dataset = data_module(collate_fn=collate_func,
                              dataset_type=dataset_type)
        dataset.prepare_data()
        dataset.setup()