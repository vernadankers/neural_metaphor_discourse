"""
Data functions for the metaphor gating project.

Create a TextDataset object to gather data samples and turn them into batches.
"""

import csv
import codecs
import torch

from collections import defaultdict

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import LongTensor as LT, FloatTensor as FT
from pytorch_pretrained_bert import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class Batch:
    """Batch object."""

    def __init__(self, sentences, lengths, labels, mask, context,
                 context_lengths, context_masks, tokens=None,
                 context_tokens=None, mapping=None, bert_labels=None):
        """Initialise Batch object.

        Args:
            src (list of str): source sentence
            tgt (list of str): target sentence
            src_tokens (LongTensor): BERTTOKENIZER token ids
            tgt_tokens (LongTensor): BERTTOKENIZER token ids
            mask (LongTensor): mask corresponding to src_tokens
        """
        # Standard focus sentence tensors
        self.sentences = sentences
        self.lengths = torch.LongTensor(lengths)
        self.mask = torch.FloatTensor(mask)

        # Context tensors
        self.context = context
        self.context_lengths = [torch.LongTensor(x) for x in context_lengths]
        self.context_masks = [torch.FloatTensor(x) for x in context_masks]

        # BERT specific tensors
        bert = bert_labels is not None
        self.mapping = mapping
        self.context_tokens = context_tokens if not bert else \
            [torch.LongTensor(x) for x in context_tokens]
        self.tokens = tokens if not bert else torch.LongTensor(tokens)
        self.labels = labels if bert else torch.LongTensor(labels)
        self.bert_labels = bert_labels \
            if not bert else torch.LongTensor(bert_labels)


class CustomDataset(Dataset):
    """Dataset object."""

    def __init__(self, texts, labels, context):
        self.texts = texts
        self.labels = labels
        self.context = context

    def __getitem__(self, idx):
        """Extract one sample with index idx.

        Args:
            idx (int): sample number

        Returns:
            text (list of str): words
            len (int): number of words
            label (int): 0 or 1, label
        """
        text = self.texts[idx]
        label = self.labels[idx]
        context = self.context[idx]
        return text, label, context

    def __len__(self):
        """Compute number of samples in dataset."""
        return len(self.labels)

    @staticmethod
    def bert_process_sentences(sentences):
        # Retokenise using BERT-specific tokenizer
        wp_sentences, wp_lengths = [], []
        for s in sentences:
            wp_sentence = tokenizer.convert_tokens_to_ids(["[CLS]"])
            for word in s:
                word = tokenizer.tokenize(word)
                word = tokenizer.convert_tokens_to_ids(word)
                wp_sentence.extend(word)
            wp_sentences.append(wp_sentence + tokenizer.convert_tokens_to_ids(["[SEP]"]))
            wp_lengths.append(len(wp_sentence) + 1)
        maxi = max([len(s) for s in wp_sentences])

        # Construct padded texts, compute token indices, construct mask
        tokens, mask = [], []
        for s in wp_sentences:
            mask.append([1] * len(s) + [0] * (maxi - len(s)))
            tokens.append(s + [0] * (maxi - len(s)))
        return tokens, wp_lengths, mask

    @staticmethod
    def collate_fn_bert(batch):
        sentences, labels, context = zip(*batch)

        tokens, lengths, mask = CustomDataset.bert_process_sentences(sentences)
        context_tokens, context_lengths, context_masks = [], [], []
        context = [list(x) for x in zip(*context)]
        for i, _ in enumerate(context):
            for j, snt in enumerate(context[i]):
                if not snt:
                    context[i][j] = ["[PAD]"]
        for s in context:
            if s:
                t, l, m = CustomDataset.bert_process_sentences(s)
                context_tokens.append(t)
                context_lengths.append(l)
                context_masks.append(m)

        # Retokenise using BERT-specific tokenizer
        bert_labels, mapping = [], []
        for sentence, label, length in zip(sentences, labels, lengths):
            mapping_list, target_list = [], []
            for i, (word, t) in enumerate(zip(sentence, label)):
                word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                mapping_list.extend([i] * len(word))
                target_list.extend([t] * len(word))
            target_list = target_list + [-1.0] * (max(lengths) - length)
            bert_labels.append(target_list)
            mapping.append(mapping_list)

        batch = Batch(sentences, lengths, labels, mask, context,
                      context_lengths, context_masks, tokens, context_tokens,
                      mapping, bert_labels)
        return batch

    @staticmethod
    def collate_fn(batch):
        """Gather the batches' text, lengths, labels and masks.

        Args:
            batch (list of tuples): texts, lengths and labels

        Returns:
            custom Batch object
        """
        sentences, labels, context = zip(*batch)
        lengths = [len(s) for s in sentences]
        labels = [x + [-1] * (max(lengths) - len(x)) for x in labels]
        mask = [[1] * x + [0] * (max(lengths) - x) for x in lengths]

        ctx_lengths, ctx_masks = [], []
        context = [list(x) for x in zip(*context)]
        for i, ctx_sentences in enumerate(context):
            l = [max(1, len(s)) for s in ctx_sentences]
            m = [[1] * len(x) + [0] * (max(l) - len(x)) for x in ctx_sentences]
            for j, snt in enumerate(ctx_sentences):
                if not snt:
                    context[i][j] = ["<PAD>"]
            ctx_lengths.append(l)
            ctx_masks.append(m)

        batch = Batch(sentences, lengths, labels, mask, context,
                      ctx_lengths, ctx_masks)
        return batch


def get_metaphor_data(filename, batch_size, k, bert, train=False):
    """Prepare the DataLoader and TextDataset objects to load data with.

    Args:
        filename (str): filename of VUA metaphor dataset
        batch_size (int): batch size
        train_steps (int): maximum number of batches
    Returns:
        DataLoader
    """
    data = []
    corpus = defaultdict(lambda: defaultdict(list))

    with codecs.open(filename, encoding="latin-1") as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sentence = line[2].replace("M_", "").replace("L_", "").split()
            snt_id = int(line[1].replace("a", ""))
            if sentence:
                corpus[line[0]][snt_id] = sentence

    with codecs.open(filename, encoding="latin-1") as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sentence = line[2].replace("M_", "").replace("L_", "").split()
            if sentence:
                label_seq = [0 if "L_" in w else 1 if "M_" in w else -1
                             for w in line[2].split()]
                assert len(label_seq) == len(sentence)
                snt_id = int(line[1].replace("a", ""))
                context = []
                for i in range(snt_id - k, snt_id + 1 + k):
                    context.append(corpus[line[0]][i])
                data.append((sentence, label_seq, context))

    snts, labels, context = zip(*data)

    # Get dataloaders to give us batches
    dataset = CustomDataset(list(snts), list(labels), list(context))
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)

    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=CustomDataset.collate_fn if not bert
        else CustomDataset.collate_fn_bert
    )


def get_vocab(meta_train, meta_dev, meta_test):
    """Prepare the DataLoader and TextDataset objects to load data with.

    Args:
        filename (str): filename of VUA metaphor dataset
        filename (str): filename of VUA metaphor dataset
        filename (str): filename of VUA metaphor dataset

    Returns:
        vocab (list): list of vocabulary words
        sentences (list): all sentences to load, sorted
    """
    vocab = set()
    sentences = []

    for filename in [meta_train, meta_dev, meta_test]:
        with codecs.open(filename, encoding="latin-1") as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                sentence = line[2].replace("M_", "").replace("L_", "").split()
                if sentence:
                    vocab.update(set(sentence))
                    sentences.append(sentence)
    return list(vocab), sorted(sentences, key=lambda item: len(item))
