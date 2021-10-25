def forward(self, batch_qs, batch_qs_pos, batch_as, batch_as_pos):
    batch_size = len(batch_qs)

    print('-' * 100)
    print('batch_qs shape:', batch_qs.shape)
    print('batch_qs_pos shape:', batch_qs_pos.shape)
    print('batch_as shape:', batch_as_pos.shape)
    print('batch_as_pos shape:', batch_as_pos.shape)
    print('-' * 100)
    print('Sample from batch_qs:')
    print(batch_qs[0])
    print('Decoded the above:')
    print(np_decode_string(batch_qs[0].numpy()))
    print('Sample from batch_qs_pos:')
    print(batch_qs_pos[0])
    print('-' * 100)
    print('Sample from batch_as:')
    print(batch_as[0])
    print('Decoded the above:')
    print(np_decode_string(batch_as[0].numpy()))
    print('Sample from batch_as_pos:')
    print(batch_as_pos[0])

    print('-' * 100)
    batch_qs = torch.transpose(batch_qs, 0, 1)
    print('Transposed batch_qs:', batch_qs.shape)
    batch_qs = nn.functional.one_hot(batch_qs, self.vocab_sz)
    print('One hot encoded batch_qs with vocab_sz', batch_qs.shape)
    print('Sample from batch_qs after one hot:')
    print(batch_qs[0])
    print('-' * 100)

    batch_as = batch_as[:, 1:]
    print('Sliced batch_as:', batch_as.shape)
    batch_as = torch.transpose(batch_as, 0, 1)
    print('Transformed batch_as:', batch_as.shape)
    batch_as = nn.functional.one_hot(batch_as, self.vocab_sz)
    print('One hot encoded batch_as with vocab_sz', batch_as.shape)
    print('Sample from batch_as after one hot:')
    print(batch_as[0])
    print('-' * 100)

    batch_as = batch_as.float()

    batch_qs = batch_qs.float()  # (max_q_sz, batch_sz, vocab_sz)

    hidden_state = Variable(
        torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float))
    print('hidden state shape:', hidden_state.shape)
    cell_state = Variable(
        torch.zeros(1, batch_size, self.num_hidden, dtype=torch.float))
    print('cell state shape:', cell_state.shape)
    output_seq = torch.empty(
        (self.max_answer_sz - 1, batch_size, self.vocab_sz))
    print('output_seq shape:', output_seq.shape)
    thinking_input = torch.zeros(1,
                                 batch_size,
                                 self.vocab_sz,
                                 dtype=torch.float)
    print('thinking_input shape:', thinking_input.shape)
    if torch.cuda.is_available():
        hidden_state = hidden_state.cuda()
        cell_state = cell_state.cuda()
        output_seq = output_seq.cuda()
        thinking_input = thinking_input.cuda()
        batch_as = batch_as.cuda()
        batch_qs = batch_qs.cuda()
    print('-' * 100)
    print('Input question phase')
    # Input question phase
    self.lstm.flatten_parameters()
    for t in range(self.max_question_sz):
        outputs, (hidden_state,
                  cell_state) = self.lstm(batch_qs[t].unsqueeze(0),
                                          (hidden_state, cell_state))
        print(f'size of batch_qs[{t}]: {batch_qs[t].shape}')
        print(f'Unsqueezed batch_qs[{t}]: {batch_qs[t].unsqueeze(0).shape}')
        print('outputs shape:', outputs.shape)
        print('hidden_state shape:', hidden_state.shape)
        print('cell_state shape:', cell_state.shape)
        print('-' * 50)

    print('-' * 100)
    print('Extra 15 Computational Steps')
    # Extra 15 Computational Steps
    for t in range(15):
        outputs_junk, (hidden_state,
                       cell_state) = self.lstm(thinking_input,
                                               (hidden_state, cell_state))
        print('outputs_junk shape:', outputs_junk.shape)
        print('hidden_state shape:', hidden_state.shape)
        print('cell_state shape:', cell_state.shape)
        print('-' * 50)
    print('-' * 100)
    print('Answer generation phase')
    # Answer generation phase, need to input correct answer as hidden/cell state, find what to put in input
    for t in range(self.max_answer_sz - 1):
        if t == 0:
            out = self.tgt_word_prj(outputs)
            print('out of tgt_words_prj:', out.shape)
            output_seq[t] = out
            char = output_seq[t].clone().unsqueeze(0)
            print(f'output_seq[{t}] unsqueezed:', char.shape)
            outputs, (hidden_state,
                      cell_state) = self.lstm(char, (hidden_state, cell_state))
            print('outputs shape:', outputs.shape)
            print('hidden_state shape:', hidden_state.shape)
            print('cell_state shape:', cell_state.shape)
            print('-' * 50)
        else:
            output_seq[t] = self.tgt_word_prj(outputs)
            print(f'output_seq[{t}] of tgt_words_prj:', output_seq[t].shape)
            outputs, (hidden_state,
                      cell_state) = self.lstm(batch_as[t].unsqueeze(0),
                                              (hidden_state, cell_state))
            print('outputs shape:', outputs.shape)
            print('hidden_state shape:', hidden_state.shape)
            print('cell_state shape:', cell_state.shape)
            print('-' * 50)
    print('-' * 100)
    print('output_seq size:', output_seq.size())
    batch_ans_size = output_seq.size(2)  # batch_size x answer_length
    print('output_seq size(2) i.e. batch_ans_size:', output_seq.size(2))
    print('reshaped output_seq:', output_seq.reshape(-1, batch_ans_size).shape)
    return output_seq.reshape(-1, batch_ans_size)
