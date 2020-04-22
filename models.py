"""
Models for the metaphor & discourse project.

Create the neural network that we will use for metaphor prediction.
- ELMo-LSTM:
    - an embedding layer, that maps words to ELMo embeddings
    - (a) recurrent layer(s), that are bidirectional LSTMs
    - a discourse module
    - a classification layer including a softmax activation

- BERT:
    - BERT-base-uncased
    - a discourse module
    - a classification layer including a softmax activation
"""
import logging
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.nn.util import sort_batch_by_length
from torchnlp.word_to_vector import GloVe
from transformers import BertModel
from attention import GeneralAttention, HierarchicalAttention

logging.getLogger("allennlp").setLevel(logging.WARNING)

SITE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_"
OPTS = SITE + "2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHTS = SITE + "2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class MetaphorModel(nn.Module):
    """Metaphor detection model."""

    def __init__(self, vocab, sentences, model="elmo", attention="vanilla", k=0):
        super().__init__()
        self.name = f"model={model}_attention={attention}_k={k}"
        self.k = k
        self.bert = model == "bert"

        if self.bert:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.train()
            hidden_size = 768
            double = hidden_size
            self.dropout_on_input_to_linear_layer = nn.Dropout(0.1)
        else:
            hidden_size = 128
            self.cached = dict()
            self.init_elmo(vocab, sentences)
            self.glove = GloVe()

            self.lstm1 = nn.LSTM(
                input_size=1324, hidden_size=hidden_size,
                batch_first=True, bidirectional=True
            )
            double = hidden_size * 2
            self.dropout_on_input_to_LSTM = nn.Dropout(0.5)
            self.dropout_on_input_to_linear_layer = nn.Dropout(0.3)

        # Initialise parameters for classification & attention layers
        self.output_projection = nn.Linear(double * (2 if k > -1 else 1), 2)
        if attention == "general":
            self.attention = GeneralAttention(double)
        else:
            self.attention = HierarchicalAttention(double, self.bert)

    def forward(self, inputs, lengths, mask, context, ctx_lengths, ctx_masks):
        """
        Forward pass of ELMo-LSTM model for emotion prediction.

        Args:
            inputs: list of lists, containing tokenised sequences of strings
            lengths: PyTorch LongTensor of sequence lengths
            mask: PyTorch LongTensor of zeros and ones

        Returns:
            predictions: batch_size x max_sent_length of float predictions
        """
        # Encode data using ELMo + LSTM / BERT
        focus_output, mask = self.encode(inputs, lengths, mask)

        if self.k > -1:
            if self.k > 0:
                # Run context sentences through the LSTM
                ctx_output = []
                for i, (s, l, m) in enumerate(zip(context, ctx_lengths, ctx_masks)):
                    o, m = self.encode(s, l, m)
                    ctx_output.append(o)
                    ctx_masks[i] = m
            else:
                ctx_output = [focus_output]
                ctx_masks = [mask]

            ctx_vector = self.attention(focus_output, ctx_output, ctx_masks)
            ctx_vector = ctx_vector.repeat(1, focus_output.shape[1], 1)
            focus_output = torch.cat((focus_output, ctx_vector), dim=-1)

        # Final classification followed by sigmoid
        output = self.dropout_on_input_to_linear_layer(focus_output)
        return torch.log_softmax(self.output_projection(output), dim=-1)

    def retrieve_embs(self, sentences, lengths):
        """
        Retrieve the ELMo embeddings corresponding to the words given as input.

        Args:
            sentences (list of lists of strings): sentences
            lengths (LongTensor): lengths of sentences

        Returns:
            embs (FloatTensor): containing word embeddings
        """
        m = torch.max(lengths).item()
        batch = []
        for s, l in zip(sentences, lengths.tolist()):
            elmo = (self.cached[' '.join(s)], torch.zeros((m - l, 1024)))
            glove = self.glove[s + ["<pad>"] * (m - l)]
            batch.append(torch.cat((torch.cat(elmo, dim=0), glove), dim=-1))
        return torch.stack(batch, dim=0).cuda()

    def encode(self, inputs, lengths, mask):
        if not self.bert:
            embeddings = self.retrieve_embs(inputs, lengths).detach()
            embedded_input = self.dropout_on_input_to_LSTM(embeddings)
            lengths = lengths.cuda()
            lstm_output = self.run_lstm(self.lstm1, embedded_input, lengths)
            return lstm_output, mask.cuda()
        encoding = self.bert_model(inputs.cuda(), attention_mask=mask.cuda())[0]
        return encoding[:, 1:-1], mask[:, 1:-1].cuda()

    def init_elmo(self, vocab, sentences):
        """
        Cache ELMo embeddings before training starts.

        Args:
            vocab: list of lists, containing tokenised sequences of strings
            sentences: PyTorch LongTensor of sequence lengths
        """
        elmo = Elmo(OPTS, WEIGHTS, 1, dropout=0, vocab_to_cache=vocab).cuda()

        logging.info("Preloading ELMo embeddings.")
        for i in range(0, len(sentences), 64):
            idx = batch_to_ids(sentences[i:i + 64]).cuda()
            emb = elmo(idx)["elmo_representations"][0].cpu()
            for j, s in enumerate(sentences[i:i + 64]):
                self.cached[' '.join(s)] = emb[j, :len(s)]

        idx = batch_to_ids([["<PAD>"]]).cuda()
        emb = elmo(idx)["elmo_representations"][0].cpu()
        self.cached["<PAD>"] = emb[0, :1]

    @staticmethod
    def run_lstm(lstm, inputs, lengths):
        """
        Run inputs through a LSTM.

        Args:
            lstm (LSTM): LSTM to use
            inputs (FloatTensor): word embeddings
            lengths (LongTensor): vector with sentence lengths
        """
        inputs, lengths, unsort_idx, _ = sort_batch_by_length(inputs, lengths)
        inputs = pack_padded_sequence(inputs, lengths.data.tolist(), batch_first=True)
        lstm.flatten_parameters()
        packed_sorted_output, _ = lstm(inputs)
        sort, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        return sort[unsort_idx]
