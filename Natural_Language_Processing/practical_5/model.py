

import json
import operator
import os
import random
from functools import reduce
from io import open
from queue import PriorityQueue

import torch
import torch.nn as nn
from torch import optim

from utils.util import BeamSearchNode

from . import decoder, encoder, policy

SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


class Model(nn.Module):
    def __init__(
        self, args, input_lang_index2word, output_lang_index2word,
        input_lang_word2index, output_lang_word2index
    ):
        super(Model, self).__init__()
        self.args = args
        self.max_len = args.max_len

        self.output_lang_index2word = output_lang_index2word
        self.input_lang_index2word = input_lang_index2word

        self.output_lang_word2index = output_lang_word2index
        self.input_lang_word2index = input_lang_word2index

        self.hid_size_enc = args.hid_size_enc
        self.hid_size_dec = args.hid_size_dec
        self.hid_size_pol = args.hid_size_pol

        self.emb_size = args.emb_size
        self.db_size = args.db_size
        self.bs_size = args.bs_size
        self.cell_type = args.cell_type
        if 'bi' in self.cell_type:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.depth = args.depth
        self.use_attn = args.use_attn
        self.attn_type = args.attention_type

        self.dropout = args.dropout
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.teacher_forcing_ratio = args.teacher_ratio
        self.vocab_size = args.vocab_size
        self.epsln = 10E-5

        torch.manual_seed(args.seed)
        self.build_model()
        self.getCount()
        try:
            assert self.args.beam_width > 0
            self.beam_search = True
        except:
            self.beam_search = False

        self.global_step = 0

    def cuda_(self, var):
        return var.cuda() if self.args.cuda else var

    def build_model(self):
        self.encoder = encoder.EncoderRNN(
            len(self.input_lang_index2word), self.emb_size, self.hid_size_enc,
            self.cell_type, self.depth, self.dropout).to(self.device)

        if self.args.policy == 'softmax':
            self.policy = policy.SoftmaxPolicy(
                self.hid_size_pol, self.hid_size_enc, self.db_size, self.bs_size
            ).to(self.device)
        else:
            self.policy = policy.DefaultPolicy(
                self.hid_size_pol, self.hid_size_enc, self.db_size, self.bs_size
            ).to(self.device)

        if self.use_attn:
            if self.attn_type == 'bahdanau':
                self.decoder = decoder.SeqAttnDecoderRNN(
                    self.emb_size, self.hid_size_dec,
                    len(self.output_lang_index2word), self.cell_type,
                    self.dropout, self.max_len).to(self.device)
        else:
            self.decoder = decoder.DecoderRNN(
                self.emb_size, self.hid_size_dec,
                len(self.output_lang_index2word), self.cell_type, self.dropout
            ).to(self.device)

        if self.args.mode == 'train':
            self.gen_criterion = nn.NLLLoss(
                ignore_index=3, reduction='elementwise_mean')  # logsoftmax is done in decoder part
            self.setOptimizers()

    def train(self, input_tensor, input_lengths, target_tensor, target_lengths,
             db_tensor, bs_tensor, dial_name=None):
        proba, _, decoded_sent = self.forward(
            input_tensor, input_lengths, target_tensor, target_lengths,
            db_tensor, bs_tensor)

        proba = proba.view(-1, self.vocab_size)
        self.gen_loss = self.gen_criterion(proba, target_tensor.view(-1))

        self.loss = self.gen_loss
        self.loss.backward()
        grad = self.clipGradients()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.loss.item(), 0, grad

    def setOptimizers(self):
        self.optimizer_policy = None
        if self.args.optim == 'sgd':
            self.optimizer = optim.SGD(
                lr=self.args.lr_rate,
                params=[x for x in self.parameters() if x.requires_grad],
                weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adadelta':
            self.optimizer = optim.Adadelta(
                lr=self.args.lr_rate,
                params=[x for x in self.parameters() if x.requires_grad],
                weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adam':
            self.optimizer = optim.Adam(
                lr=self.args.lr_rate,
                params=[x for x in self.parameters() if x.requires_grad],
                weight_decay=self.args.l2_norm)

    def forward(
        self, input_tensor, input_lengths, target_tensor, target_lengths,
        db_tensor, bs_tensor):
        """Given the user sentence, user belief state and database pointer,
        encode the sentence, decide what policy vector construct and
        feed it as the first hidden state to the decoder."""

        # for fixed encoding this is zero so it does not contribute
        batch_size, seq_len = input_tensor.size()
        _, target_length = target_tensor.size()

        # TODO TASK e) build schema of the pipeline.
        # You should use the Encoder network and Policy.
        # Check what these classes need as input in the forward function.
        # All necessary variables are prepared for you already.
        encoder_outputs, encoder_hidden = None, None
        decoder_output, decoder_hidden = None, None
        decoder_input = None

        # YOUR CODE STARTS HERE:
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)
        decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

        # YOUR CODE ENDS HERE.

        # GENERATOR
        # Teacher forcing: Feed the target as the next input
        decoder_input = torch.LongTensor(
            [[SOS_token] for _ in range(batch_size)], device=self.device)
        proba = torch.zeros(batch_size, target_length, self.vocab_size)  # [B,T,V]
        for t in range(target_length):
            # YOUR CODE STARTS HERE:
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[:, [t]]
            # YOUR CODE ENDS HERE.
            
            proba[:, t, :] = decoder_output

        decoded_sent = None

        return proba, None, decoded_sent


    def predict(self, input_tensor, input_lengths, target_tensor,
                target_lengths, db_tensor, bs_tensor):
        with torch.no_grad():
            # ENCODER
            encoder_outputs, encoder_hidden = self.encoder(
                input_tensor, input_lengths)

            # POLICY
            decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

            # GENERATION
            decoded_words = self.decode(
                target_tensor, decoder_hidden, encoder_outputs)

        return decoded_words, 0

    def decode(self, target_tensor, decoder_hidden, encoder_outputs):
        decoder_hiddens = decoder_hidden

        if self.beam_search:
            decoded_sentences = []
            for idx in range(target_tensor.size(0)):
                if isinstance(decoder_hiddens, tuple):  # LSTM case
                    decoder_hidden = (
                        decoder_hiddens[0][:,idx, :].unsqueeze(0),
                        decoder_hiddens[1][:,idx, :].unsqueeze(0))
                else:
                    decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

                # Beam start
                self.topk = 1
                endnodes = []  # stored end nodes
                number_required = min(
                    (self.topk + 1), self.topk - len(endnodes))
                decoder_input = torch.LongTensor(
                    [[SOS_token]], device=self.device)

                # starting node hidden vector, prevNode, wordid, logp, leng,
                node = BeamSearchNode(
                    decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()  # start the queue
                nodes.put((-node.eval(None, None, None, None), node))

                # start beam search
                qsize = 1
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break

                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_hidden = n.h

                    if n.wordid.item() == EOS_token and n.prevNode != None:  # its not empty
                        endnodes.append((score, n))
                        # if reach maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_output)

                    log_prob, indexes = torch.topk(
                        decoder_output, self.args.beam_width)
                    nextnodes = []

                    for new_k in range(self.args.beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(
                            decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval(None, None, None, None)
                        nextnodes.append((score, node))

                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))

                    # increase qsize
                    qsize += len(nextnodes)

                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for n in range(self.topk)]

                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_words = utterances[0]
                decoded_sentence = [
                    self.output_index2word(str(ind.item()))
                    for ind in decoded_words]
                #print(decoded_sentence)
                decoded_sentences.append(' '.join(decoded_sentence[1:-1]))

            return decoded_sentences

        else:  # GREEDY DECODING
            decoded_sentences = self.greedy_decode(
                decoder_hidden, encoder_outputs, target_tensor)
            return decoded_sentences

    def greedy_decode(self, decoder_hidden, encoder_outputs, target_tensor):
        decoded_sentences = []
        batch_size, seq_len = target_tensor.size()
        decoder_input = torch.LongTensor(
            [[SOS_token] for _ in range(batch_size)], device=self.device)

        decoded_words = torch.zeros((batch_size, self.max_len))
        for t in range(self.max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)

            decoded_words[:, t] = topi
            decoder_input = topi.detach().view(-1, 1)

        for sentence in decoded_words:
            sent = []
            for ind in sentence:
                if self.output_index2word(
                    str(int(ind.item()))) == self.output_index2word(str(EOS_token)):
                    break
                sent.append(self.output_index2word(str(int(ind.item()))))
            decoded_sentences.append(' '.join(sent))

        return decoded_sentences

    def clipGradients(self):
        grad = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.args.clip)
        return grad

    def saveModel(self, iter):
        print('Saving parameters..')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.encoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.enc')
        torch.save(self.policy.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.pol')
        torch.save(self.decoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.dec')

        with open(self.model_dir + self.model_name + '.config', 'w') as f:
            f.write(str(json.dumps(vars(self.args), ensure_ascii=False, indent=4)))

    def loadModel(self, iter=0):
        print('Loading parameters of iter %s ' % iter)
        self.encoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.enc'))
        self.policy.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.pol'))
        self.decoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.dec'))

    def input_index2word(self, index):
        if index in self.input_lang_index2word:
            return self.input_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def output_index2word(self, index):
        if index in self.output_lang_index2word:
            return self.output_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def input_word2index(self, index):
        if index in self.input_lang_word2index:
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        if index in self.output_lang_word2index:
            return self.output_lang_word2index[index]
        else:
            return 2

    def getCount(self):
        learnable_parameters = [
            p for p in self.parameters() if p.requires_grad]
        param_cnt = sum(
            [reduce((lambda x, y: x * y), param.shape)
            for param in learnable_parameters])

    def printGrad(self):
        learnable_parameters = [
            p for p in self.parameters() if p.requires_grad]
        for idx, param in enumerate(learnable_parameters):
            print(param.grad, param.shape)
