TRANSFORMER = "transformer"
SIMPLE_LSTM = "simLSTM"
ATTENTIONAL_LSTM = "attLSTM"

FULL = 'full'
MINI = 'mini'

MODELS = [TRANSFORMER, SIMPLE_LSTM, ATTENTIONAL_LSTM]
DATASETS = [FULL, MINI]

# Math Dataset constants (from paper)
VOCAB_SZ = 95  # input chars are selected from basic ASCII chars
MAX_QUESTION_SZ = 162  # questions have less than 160 chars (!)
MAX_ANSWER_SZ = 32  # answers have less than 30 chars (!)

PAD = 0
UNK = 1
BOS = 2
EOS = 3

DATA_ID = '1Q0VGoff77nr7JO_hMK9YWlBammMCW0IX'