import torch
import torch.nn as nn
from torch.autograd import Variable
from deep_math.constants import VOCAB_SZ, MAX_ANSWER_SZ, MAX_QUESTION_SZ


class SimpleLSTM(nn.Module):

    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.num_hidden = 2048
        self.vocab_sz = VOCAB_SZ
        self.max_answer_sz = MAX_ANSWER_SZ
        self.max_question_sz = MAX_QUESTION_SZ

        self.lstm = nn.LSTM(VOCAB_SZ, self.num_hidden, 1)
        self.tgt_word_prj = nn.Linear(self.num_hidden, VOCAB_SZ, bias=False)

    def forward(self, x):

        batch_size = len(x)

        x = torch.transpose(x, 0, 1)
        x = nn.functional.one_hot(x, self.vocab_sz)

        x = x.float()  # (max_q_sz, batch_sz, vocab_sz)

        # type_as(x) is required for setting the device (cuda or cpu)
        # https://forums.pytorchlightning.ai/t/training-fails-but-found-at-least-two-devices-cuda-0-and-cpu/694

        hidden_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden,
                        dtype=torch.float)).type_as(x)
        cell_state = Variable(
            torch.zeros(1, batch_size, self.num_hidden,
                        dtype=torch.float)).type_as(x)
        output_seq = torch.empty(
            (self.max_answer_sz - 1, batch_size, self.vocab_sz)).type_as(x)
        thinking_input = torch.zeros(1,
                                     batch_size,
                                     self.vocab_sz,
                                     dtype=torch.float).type_as(x)

        # Input question phase
        self.lstm.flatten_parameters()
        for t in range(self.max_question_sz):
            outputs, (hidden_state,
                      cell_state) = self.lstm(x[t].unsqueeze(0),
                                              (hidden_state, cell_state))
        # Extra 15 Computational Steps
        for t in range(15):
            _, (hidden_state,
                cell_state) = self.lstm(thinking_input,
                                        (hidden_state, cell_state))

        # Answer generation phase, need to input correct answer as hidden/cell state, find what to put in input
        for t in range(self.max_answer_sz - 1):
            output_seq[t] = self.tgt_word_prj(outputs)
            char = output_seq[t].clone().unsqueeze(0)
            outputs, (hidden_state,
                      cell_state) = self.lstm(char, (hidden_state, cell_state))

        batch_ans_size = output_seq.size(2)  # batch_size x answer_length
        return output_seq.reshape(-1, batch_ans_size)