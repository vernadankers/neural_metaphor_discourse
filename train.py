"""Training and evaluation functions for the metaphor and discourse project."""

import gc
import copy
import logging
import numpy as np
import torch

from torch.nn import NLLLoss
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_cosine_schedule_with_warmup, AdamW


def train(model, metaphor_train, metaphor_dev, epochs, lr):
    """
    Train MetaphorModel on Metaphor data.

    Args:
        model (nn.Module): initialised model, untrained
        metaphor_train (DataLoader): object containing metaphor training data.
        metaphor_dev (DataLoader): object containing metaphor validation data.
        lr (float): learning rate for optimiser

    Returns:
        best_model: state_dict of the best model according to validation data.
    """

    # Optimiser
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimiser = AdamW(trainable_parameters, lr=lr, eps=1e-8)

    # Learning rate scheduling
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=int(epochs * 0.1 * len(metaphor_train)),
        num_training_steps=len(metaphor_train) * epochs
    )

    best_score, best_model, losses = -1, None, []
    threshold = 250 if model.bert else 100

    for x in range(epochs):
        # Setup weighted loss function
        weights = torch.FloatTensor(
            [min(0.1 + (0.05 * x), 0.3), max(0.7, 0.9 - (0.05 * x))]
        )
        loss_fn = NLLLoss(weight=weights, ignore_index=-1)
        losses = []
        for i, batch in enumerate(metaphor_train):

            # Forward pass through the model
            model.train()
            optimiser.zero_grad()
            output = model(
                batch.tokens if model.bert else batch.sentences, batch.lengths,
                batch.mask, batch.context_tokens if model.bert else batch.context,
                batch.context_lengths, batch.context_masks
            ).cpu()

            # Compute the loss value, backward pass
            labels = batch.labels if not model.bert else batch.bert_labels
            loss = loss_fn(output.view(-1, 2), labels.view(-1))
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            torch.cuda.empty_cache()

            # Evaluation loop
            if (x * len(metaphor_train) + i + 1) % threshold == 0:
                logging.info(f"Metaphor Loss: {np.mean(losses):.3f}")
                score_dev = evaluate_metaphor(model, metaphor_dev)
                if score_dev > best_score:
                    best_score = score_dev
                    clean_object_from_memory(best_model)
                    best_model = copy.deepcopy(model.state_dict())
                torch.cuda.empty_cache()
    return best_model


def evaluate_metaphor(model, dataloader, dataset_name="validation", output=None):
    """
    Evaluation metaphor model on VU Amsterdam metaphor test data.

    Args:
        model (nn.Module): model to evaluate
        dataloader (DataLoader): contains the validation batches
        dataset_name: validation / test
        bert (bool): filename of development set
    """
    model.eval()

    pairs = []
    trace = []
    for batch in dataloader:
        # Run model in inference mode
        if not model.bert:
            # ELMo-LSTM model
            outputs = model(
                batch.sentences, batch.lengths, batch.mask, batch.context,
                batch.context_lengths, batch.context_masks
            )

            for i, (prediction, length) in enumerate(zip(outputs, batch.lengths)):
                prediction = torch.argmax(prediction[:length], dim=-1).cpu().tolist()
                target = batch.labels[i, :length].cpu().tolist()
                pairs.extend([(t, p) for t, p in \
                    zip(target, prediction) if t != -1])

                # Save a trace for analysis purposes
                trace.append((
                    batch.sentences[i], target, prediction,
                    [x[i] for x in batch.context], model.attention.weights[i]
                ))
        else:
            # BERT requires remapping the outputs to the original tokens
            outputs = model(
                batch.tokens, batch.lengths, batch.mask, batch.context_tokens,
                batch.context_lengths, batch.context_masks
            )
            for i, (prediction, length) in enumerate(zip(outputs, batch.lengths)):
                prediction = torch.argmax(prediction[:length], dim=-1).cpu().tolist()
                prediction = transform(prediction, batch.mapping[i])
                pairs.extend([(t, p) for t, p in \
                    zip(batch.labels[i], prediction) if t != -1])

                # Save a trace for analysis purposes
                trace.append((
                    batch.sentences[i], batch.labels[i], prediction,
                    [x[i] for x in batch.context], model.attention.weights[i]
                ))

    tgt, prd = zip(*pairs)
    p, r, f1, _ = precision_recall_fscore_support(tgt, prd, average="binary")
    logging.info(f"{dataset_name}, P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}")

    # If the output filename of the trace is specified, save the traces
    # to file
    if output is not None:
        with codecs.open(output, 'w', encoding="utf-8") as f:
            ctx_string = "\t".join([str(i) for i in range(-model.k, model.k + 1)])
            f.write(f"sentences\ttarget\tprediction\t{ctx_string}\tattention\n")
            for sentence, label, prediction, context, weights in trace:
                sentence = " ".join(sentence)
                label = " ".join([str(x) for x in label])
                prediction = " ".join([str(x) for x in prediction])
                context = "\t".join([" ".join(x) for x in context])
                f.write(f"{sentence}\t{label}\t{prediction}\t{context}\t{weights}\n")
    return f1


def clean_object_from_memory(obj):
    """
    Clean Pytorch object from memory.

    Args:
        obj: Pytorch object
    """
    del obj
    gc.collect()
    torch.cuda.empty_cache()


def transform(prediction_sent, ids_sent):
    """
    Map the BERT predictions per sentence piece back to the original words.

    Args:
        prediction_sent (list): prediction per sentence piece
        ids_sent (list): index of sentence pieces that map back to words
    Returns:
        list containing predictions for the original words
    """
    mapping = {i: None for i in range(max(ids_sent) + 1)}
    for prediction_word, index in zip(prediction_sent, ids_sent):
        if mapping[index] is None or mapping[index] < prediction_word:
            mapping[index] = prediction_word
    return [mapping[i] for i in range(max(ids_sent) + 1)]
