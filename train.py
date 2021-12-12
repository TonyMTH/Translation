import random
import model as md
import preprocess as pr
from parameters import *

# Download Dataset
print("1. Downloading Dataset ....")
pr.download_dataset(dataset_url, save_zip_loc, save_data_folder)
print("Download Complete!\n")

# Prepare Data
print("2. Preparing Data ....\n")
input_lang, output_lang, pairs = pr.prepareData(from_, to_, MAX_LENGTH, eng_prefixes, data_file_folder, True)
print(random.choice(pairs))


# Training
print("3. Training ....\n")
encoder = encoder(input_lang)
attn_decoder = attn_decoder(output_lang)

pr.trainIters(encoder, attn_decoder, n_iters, input_lang, output_lang, SOS_token, EOS_token, device, encoder_optimizer
              , decoder_optimizer, criterion, teacher_forcing_ratio, MAX_LENGTH, pairs, saved_model_device,
               encoder_path, decoder_path, print_every=print_every, plot_every=plot_every)

# Random Evaluation
print("4. Random Evaluation ....\n")
pr.evaluateRandomly(encoder, attn_decoder, pairs, input_lang, output_lang, EOS_token, SOS_token, device, MAX_LENGTH, n=10)

# Evaluation
print("5. Evaluation ....\n")
sentence = "elle a cinq ans de moins que moi ."
pr.evaluateAndShowAttention(input_lang, encoder, attn_decoder, output_lang, EOS_token, SOS_token, device,
                             sentence, MAX_LENGTH)
sentence = "elle est trop petit ."
pr.evaluateAndShowAttention(input_lang, encoder, attn_decoder, output_lang, EOS_token, SOS_token, device,
                             sentence, MAX_LENGTH)
sentence = "je ne crains pas de mourir ."
pr.evaluateAndShowAttention(input_lang, encoder, attn_decoder, output_lang, EOS_token, SOS_token, device,
                             sentence, MAX_LENGTH)
sentence = "c est un jeune directeur plein de talent ."
pr.evaluateAndShowAttention(input_lang, encoder, attn_decoder, output_lang, EOS_token, SOS_token, device,
                             sentence, MAX_LENGTH)
if __name__ == '__main__':
    pass
