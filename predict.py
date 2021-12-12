# Import libraries
import pickle

import preprocess as pr
from parameters import *


class Translate:
    def __init__(self, encoder_path, decoder_path):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

    def predict(self, sentence):
        # Load Model
        input_lang, output_lang, pairs = pr.prepareData(from_, to_, MAX_LENGTH, eng_prefixes, data_file_folder, True)
        with open(self.encoder_path, "rb") as f:
            encoder = pickle.load(f)
        with open(self.decoder_path, "rb") as f:
            decoder = pickle.load(f)

        out = pr.evaluate(encoder, decoder, input_lang, output_lang, EOS_token, SOS_token, torch.device("cpu"), sentence, MAX_LENGTH)
        return ' '.join(out[0][:-1])


if __name__ == '__main__':
    sentence = "elle a cinq ans de moins que moi ."
    out = Translate(encoder_path,decoder_path).predict(sentence)
    print(out)
