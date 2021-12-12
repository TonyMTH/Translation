import copy
import math
import os
import pickle
import random
import re
import time
import zipfile

import requests
import torch
import unicodedata

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def download_dataset(url, save_file_name, save_folder):
    if not os.path.isfile(save_file_name):
        with open(save_file_name, "wb") as target:
            target.write(requests.get(url).content)
        with zipfile.ZipFile(save_file_name, 'r') as zip_ref:
            zip_ref.extractall(save_folder)
        os.remove(save_file_name)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, file_path, reverse=False):
    # print("Reading lines...")

    # Read the file and split into lines
    lines = open(file_path + '%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, MAX_LENGTH, eng_prefixes):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs, MAX_LENGTH, eng_prefixes):
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH, eng_prefixes)]


def prepareData(lang1, lang2, MAX_LENGTH, eng_prefixes, file_path, reverse=False):
    # 1. Read text file and split into lines, split lines into pairs
    # 2. Normalize text, filter by length and content
    # 3. Make word lists from sentences in pairs

    input_lang, output_lang, pairs = readLangs(lang1, lang2, file_path, reverse)
    # print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, MAX_LENGTH, eng_prefixes)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    # print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # print("Counted words:")
    # print(input_lang.name, input_lang.n_words)
    # print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, EOS_token, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, EOS_token, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], EOS_token, device)
    target_tensor = tensorFromSentence(output_lang, pair[1], EOS_token, device)
    return input_tensor, target_tensor


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,
          device, SOS_token, EOS_token, teacher_forcing_ratio, saved_model_device,
          encoder_path, decoder_path):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei])
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, _, decoder_attention = decoder(decoder_input, encoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, _, decoder_attention = decoder(decoder_input, encoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()


    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, input_lang, output_lang, SOS_token, EOS_token, device, encoder_optimizer,
               decoder_optimizer, criterion, teacher_forcing_ratio, max_length, pairs, saved_model_device,
               encoder_path, decoder_path, print_every=1000, plot_every=100):
    # 1. Start a timer
    # 2. Initialize optimizers and criterion
    # 3. Create set of training pairs
    # 4. Start empty losses array for plotting

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    least_loss_avg = np.inf

    encoder_optimizer = encoder_optimizer(encoder.parameters())
    decoder_optimizer = decoder_optimizer(decoder.parameters())
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, EOS_token, device)
                      for i in range(n_iters)]

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                     max_length, device, SOS_token, EOS_token, teacher_forcing_ratio, saved_model_device,
                     encoder_path, decoder_path)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # Save best model
            if print_loss_avg <= least_loss_avg:
                least_loss_avg = print_loss_avg

                print("Pickling in progress ...")
                best_decoder = copy.deepcopy(decoder)
                best_encoder = copy.deepcopy(encoder)
                best_decoder.to(saved_model_device)
                best_encoder.to(saved_model_device)
                with open(decoder_path, "wb") as f:
                    pickle.dump(best_decoder, f)
                with open(encoder_path, "wb") as f:
                    pickle.dump(best_encoder, f)
                print("Pickling Done!")

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, input_lang, output_lang, EOS_token, SOS_token, device, sentence, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, EOS_token, device)

        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei])
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, _, decoder_attention = decoder(decoder_input, encoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, EOS_token, SOS_token, device, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, EOS_token, SOS_token,
                                            device, pair[0], max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_lang, encoder1, attn_decoder1, output_lang, EOS_token, SOS_token, device,
                             sentence, max_length):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_lang, output_lang, EOS_token, SOS_token, device,
                                        sentence, max_length)
    print('input =', sentence)
    print('output =', ' '.join(output_words))
    showAttention(sentence, output_words, attentions)
