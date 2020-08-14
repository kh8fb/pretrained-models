"""
Script for sentiment analysis prediction on a single sentence by finetuned BERT-ITPT model.
"""

from collections import OrderedDict
import torch
from transformers import BertTokenizer
from bert_model import BertForSequenceClassification, BertConfig


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
    model: BertForSequenceClassification
        Model with the loaded pretrained states
    """
    config = BertConfig(vocab_size=30522, type_vocab_size=2)
    model = BertForSequenceClassification(config, 2, [11])
    model_states = torch.load(model_path, map_location=device)
    model.load_state_dict(model_states)
    model.eval()
    return model


def prepare_input(sentence, tokenizer):
    """
    Tokenizes, truncates, and prepares the input for modeling.

    Parameters
    ----------
    sentence: str
        Input sentence to obtain sentiment from.
    tokenizer: BertTokenizer
        Tokenizer for tokenizing input.

    Returns
    -------
    input_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Encoded form of the input sentence.
    tok_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify token type for the model.
        Because sentiment analysis uses only one input, this is just a tensor of zeros.
    att_mask: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify attention masking for the model.
    """
    input_ids = torch.tensor([tokenizer.encode(sentence, padding='max_length')])
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
    tok_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
    att_mask = torch.ones(input_ids.shape, dtype=torch.int64)
    return input_ids, tok_type_ids, att_mask


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
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = load_model(model_path, device)
    input_ids, token_type_ids, attention_mask = prepare_input(sentence, tokenizer)
    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    result = torch.argmax(outputs[0], dim=-1).item()
    print(outputs[0])
    return result
