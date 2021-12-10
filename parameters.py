# Dataset
import torch
from torch import optim, nn

dataset_url = 'https://download.pytorch.org/tutorial/data.zip'
save_zip_loc = 'data/data.zip'
save_data_folder = 'data/'
data_file_folder = 'data/data/'
from_, to_ = 'eng', 'fra'

# Cleaning Data
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
teacher_forcing_ratio = 0.5
learning_rate = 0.01
n_iters = 75000
print_every = 5000
plot_every = 100
encoder_optimizer = lambda x: optim.SGD(x, lr=learning_rate)
decoder_optimizer = lambda x: optim.SGD(x, lr=learning_rate)
criterion = nn.NLLLoss()


saved_model_device = torch.device("cpu")
encoder_path = 'data/best_encoder.pt'
decoder_path = 'data/best_decoder.pt'