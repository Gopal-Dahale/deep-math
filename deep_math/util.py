from deep_math.constants import TRANSFORMER, SIMPLE_LSTM, ATTENTIONAL_LSTM
from deep_math.data.util import question_answer_to_position_batch_collate_fn, lstm_batch_collate_fn
from deep_math.models.simple_lstm import SimpleLSTM


def collate_fn(model_type):
    if model_type == TRANSFORMER:
        return question_answer_to_position_batch_collate_fn
    elif model_type == SIMPLE_LSTM:
        return lstm_batch_collate_fn
    elif model_type == ATTENTIONAL_LSTM:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid model_type {model_type}.")


def build_model(model_type):
    if model_type == TRANSFORMER:
        return build_transformer()
    elif model_type == SIMPLE_LSTM:
        return build_simple_lstm()
    elif model_type == ATTENTIONAL_LSTM:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid model_type {model_type}.")


def build_transformer():
    return None


def build_simple_lstm():
    return SimpleLSTM()