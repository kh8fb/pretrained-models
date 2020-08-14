"""
Example illustrating how to implement intermediate-gradients with finetuned XLNet Large models.

For this example, make sure to also install intermediate-gradients in your conda environment.
intermediate-gradients can be found here https://github.com/kh8fb/intermediate-gradients
"""

from captum.attr import LayerIntegratedGradients
import click
from collections import OrderedDict
import logging
from transformers import XLNetTokenizer
import torch

from intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients
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
    model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased")
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


def sequence_forward_func(inputs, model, tok_type_ids, att_mask):
    """
    Passes forward the inputs and relevant keyword arguments.

    Parameters
    ----------
    inputs: torch.tensor(1, num_ids), dtype=torch.int64
        Encoded form of the input sentence.
    tok_type_ids: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify token type for the model.
        Because sentiment analysis uses only one input, this is just a tensor of zeros.
    att_mask: torch.tensor(1, num_ids), dtype=torch.int64
        Tensor to specify attention masking for the model.

    Returns
    -------
    outputs: torch.tensor(1, 2), dtype=torch.float32
        Output classifications for the model.
    """
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)[0]
    return outputs


@click.command(help="""Run intermediate gradients on a predetermined input and baseline tensor 
for sequence classification. This example script allows you to see the shapes and behavior of the
returned gradients, as well as how to turn them into the attributions produced
by integrated gradients.""")
@click.option(
    "--model-path",
    "-m",
    required=True,
    help="Path to finetuned pytorch model file from Google Drive link.",
)
@click.option(
    "--n-steps",
    help="The number of steps used by the approximation method. Default is 50.",
    required=False,
    default=50,
)
def main(model_path, n_steps=50):
    #pylint: disable=missing-docstring, too-many-locals

    # disable warning messages for initial pretrained XLNet module.
    logging.basicConfig(level=logging.ERROR)
    n_steps = int(n_steps)

    # load the model and tokenizer
    model = load_model(str(model_path), device=torch.device("cpu"))
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

    # tokenize the sentence for classification
    sequence = """Some might not find it to totally be like Pokémon without Ash.
    But this is definitely a Pokémon movie and way better than most of their animated movies.
    The CGI nailed the looks of all the Pokémon creatures and their voices.
    The movie is charming, funny and fun as well. They did a great job introducing this world to the
    big screen. I definitely want more."""

    features = prepare_input(sequence, tokenizer)
    input_ids = features["input_ids"]
    token_type_ids = features["token_type_ids"]
    attention_mask = features["attention_mask"]

    # create a baseline of zeros in the same shape as the inputs
    baseline_ids = torch.zeros(input_ids.shape, dtype=torch.int64)

    # instance of layer intermediate gradients based upon the dummy layer representing the embeddings
    lig = LayerIntermediateGradients(sequence_forward_func, model.transformer.batch_first)
    grads, step_sizes, intermediates = lig.attribute(inputs=input_ids,
                                      baselines=baseline_ids,
                                      additional_forward_args=(model, token_type_ids, attention_mask),
                                      target=1,
                                      n_steps=n_steps)
    
    print("Shape of the returned gradients: ")
    print(grads.shape)
    print("Shape of the step sizes: ")
    print(step_sizes.shape)

    # now calculate attributions from the intermediate gradients

    # multiply by the step sizes
    scaled_grads = grads.view(n_steps, -1) * step_sizes
    # reshape and sum along the num_steps dimension
    scaled_grads = torch.sum(scaled_grads.reshape((n_steps, 1) + grads.shape[1:]), dim=0)
    # pass forward the input and baseline ids for reference
    forward_input_ids = model.transformer.word_embedding.forward(input_ids)
    forward_baseline_ids = model.transformer.word_embedding.forward(baseline_ids)
    # multiply the scaled gradients by the difference of inputs and baselines to obtain attributions
    attributions = scaled_grads * (forward_input_ids - forward_baseline_ids)
    print("Attributions calculated from intermediate gradients: ")
    print(attributions.shape)
    print(attributions)

    # compare to layer integrated gradients
    layer_integrated = LayerIntegratedGradients(sequence_forward_func, model.transformer.batch_first)
    attrs = layer_integrated.attribute(inputs=input_ids,
                                             baselines=baseline_ids,
                                             additional_forward_args=(model, token_type_ids, attention_mask),
                                             n_steps=n_steps,
                                             target=1,
                                             return_convergence_delta=False)
    print("Attributions from layer integrated gradients: ")
    print(attrs.shape)
    print(attrs)

    print("Intermediate tensor shape: ", intermediates.shape)
    print("Intermediate tensor: ", intermediates)
    

if __name__ == "__main__":
    main()
