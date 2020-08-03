"""
Script for sentiment analysis prediction on a single sentence by finetuned xlnet-base-cased-imdb model.
"""

from collections import OrderedDict
import logging
import torch
from transformers import XLNetTokenizer

from modified_xlnet import XLNetForSequenceClassification


def load_model(model_path, device):
    """
    Load the pretrained model states and prepare the model for sentiment analysis.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.
    device: torch.device
        Device to load the model on.

    Returns
    -------
    model: XLNetForSequenceClassification
        Model with the loaded pretrained states.
    """
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
    model_states = torch.load(model_path, map_location=device)
    new_model_states = OrderedDict()
    for state in model_states:
        correct_state = state[7:]
        new_model_states[correct_state] = model_states[state]
    model.load_state_dict(new_model_states)
    model.eval()
    return model


def prepare_input(sentence, tokenizer):
    """
    Tokenizes, truncates, and prepares the input for modeling.

    NOTE: Requires Transformers>=3.0.0

    Parameters
    ----------
    sentence: str
        Input sentence to obtain sentiment from.
    tokenizer: XLNetTokenizer
        Tokenizer for tokenizing input.

    Returns
    -------
    features: dict
        Keys
        ----
        input_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Tokenized sequence text.
        token_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
            Token type ids for the inputs.
        attention_mask: torch.tensor(1, num_ids), dtype=torch.int64
            Masking tensor for the inputs.
    """
    features = tokenizer([sentence], return_tensors='pt', truncation=True, max_length=512)
    return features


def get_prediction(sentence, model_path, cuda=True):
    """
    Get the model's sentiment prediction from a single sequence string.

    Parameters
    ----------
    sentence: str
        Sentence to perform sentiment analysis on.
    model_path: str
        Path to the finetuned model downloaded from the Google Drive link.
    cuda: bool
        If true, run model on cuda, otherwise run it on cpu.

    Returns
    -------
    result: int
        0 if the model predicts a negative sentiment.
        1 if the model predicts a positive sentiment.
    """
    # disable warning messages for initial pretrained XLNet module.
    logging.basicConfig(level=logging.ERROR)
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    model = load_model(model_path, device)
    features = prepare_input(sentence, tokenizer)
    outputs = model(features["input_ids"], token_type_ids=features["token_type_ids"], attention_mask=features["attention_mask"])
    result = torch.argmax(outputs[0], dim=-1).item()
    print(outputs[0])
    return result
