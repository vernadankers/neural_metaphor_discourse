"""Main functionality for the metaphor & discourse project."""

import os
import random
import logging
import argparse
import torch
import numpy as np

from data import get_metaphor_data, get_vocab
from train import train, evaluate_metaphor
from models import MetaphorModel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    logging.info(f"Seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()

    # Task independent arguments related to preprocessing or training
    group = parser.add_argument_group("model")
    group.add_argument("--model", choices=["elmo", "bert"], default="elmo",
                       help="Model to use: elmo | bert")
    group.add_argument("--attention", choices=["general", "hierarchical"],
                       default="general", help="Attention mechanism")
    group.add_argument("--seed", type=int, default=-1, help="Random seed")
    group.add_argument("--k", type=int, default=0, help="Discourse window")

    group = parser.add_argument_group("training")
    group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    group.add_argument("--epochs", type=int, default=5, help="Num. of epochs")
    group.add_argument("--batch_size", type=int, default=64, help="Bach size")
    group.add_argument("--dev", action="store_true", help="Development mode")

    group = parser.add_argument_group("data")
    group.add_argument("--meta_train", type=str,
                       default="metaphor_data/train.csv")
    group.add_argument("--meta_dev", type=str,
                       default="metaphor_data/validation.csv")
    group.add_argument("--meta_test", type=str,
                       default="metaphor_data/test.csv")
    group.add_argument("--output", type=str, default="output.tsv")
    args = vars(parser.parse_args())
    logging.info(args)

    (meta_train, meta_dev, meta_test) = \
        (args["meta_train"], args["meta_dev"], args["meta_test"])
    # Set seed to combat random effects
    set_seed(args["seed"])

    vocab, sentences = get_vocab(meta_train, meta_dev, meta_test)
    bert = args["model"] == "bert"

    # Metaphor data filenames
    if args["dev"]:
        (meta_test, meta_dev) = (meta_dev, meta_test)

    meta_train = get_metaphor_data(
        meta_train, args["batch_size"], args["k"], bert, train=True
    )
    meta_dev = get_metaphor_data(meta_dev, 8, args["k"], bert)
    meta_test = get_metaphor_data(meta_test, 8, args["k"], bert)

    # Initialise an empty model and train it.
    logging.info("Initialised vanilla metaphor model.")
    model = MetaphorModel(vocab, sentences, model=args["model"],
                          attention=args["attention"], k=args["k"]).cuda()

    # Evaluate every epoch on the validation data.
    best_model = train(model, meta_train, meta_dev,
                       epochs=args["epochs"], lr=args["lr"])
    model.load_state_dict(best_model)

    # Evaluate the trained model on test data.
    evaluate_metaphor(
        model, meta_test, "test", output=args["output"]
    )
