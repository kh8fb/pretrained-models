"""
Example illustrating how to implement intermediate-gradients with the finetuned BERT itpt-model.

For this example, make sure to also install intermediate-gradients in your conda environment.
intermediate-gradients can be found here https://github.com/kh8fb/intermediate-gradients
"""

from captum.attr import LayerIntegratedGradients
import click
from collections import OrderedDict
from transformers import BertTokenizer
from modeling_single_layer import BertForSequenceClassification, BertConfig
import torch

from intermediate_gradients.layer_intermediate_gradients import LayerIntermediateGradients


def load_deprecated_model(model_path):
    """
    Load the pretrained model states and prepare the model for sentiment analysis on CPU.
    
    This method returns a custom BertForSequenceClassification model that allows it to work
    with LayerIntegratedGradients and LayerIntermediateGradients.

    Parameters
    ----------
    model_path: str
        Path to the pretrained model states binary file.

    Returns
    -------
    model: BertForSequenceClassification
        Model with the loaded pretrained states.
    """
    config = BertConfig(vocab_size=30522, type_vocab_size=2)
    model = BertForSequenceClassification(config, 2)
    model_states = torch.load(model_path, map_location=torch.device("cpu"))
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
    input_ids = torch.tensor([tokenizer.encode(sentence, padding='max_length')], dtype=torch.int64)
    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :512]
    tok_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
    att_mask = torch.ones(input_ids.shape, dtype=torch.int64)
    return input_ids, tok_type_ids, att_mask


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
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)
    return outputs[0]


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
    n_steps = int(n_steps)

    # load the model and tokenizer
    model = load_deprecated_model(str(model_path))
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    # tokenize the sentence for classification
    sequence = """Some might not find it to totally be like Pokémon without Ash.
    But this is definitely a Pokémon movie and way better than most of their animated movies.
    The CGI nailed the looks of all the Pokémon creatures and their voices.
    The movie is charming, funny and fun as well. They did a great job introducing this world to the
    big screen. I definitely want more."""

    input_ids, token_type_ids, attention_mask = prepare_input(sequence, tokenizer)
    print(model.forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask))

    # create a baseline of zeros in the same shape as the inputs
    baseline_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
    
    #change following to intermediate gradients
    
    # create an instance of layer intermediate gradients based upon the embedding layer
    lig = LayerIntermediateGradients(sequence_forward_func, model.bert.embeddings)
    start_grads, start_step_sizes = lig.attribute(inputs=input_ids,
                                                    baselines=baseline_ids,
                                                    additional_forward_args=(model, token_type_ids, attention_mask),
                                                    n_steps=n_steps)
    
    """lig = LayerIntegratedGradients(sequence_forward_func, model.bert.embeddings)
    start_grads, start_step_sizes = lig.attribute(inputs=input_ids,
                                                baselines=baseline_ids,
                                                additional_forward_args=(model, token_type_ids, attention_mask),
                                                return_convergence_delta=True)"""
    print("Shape of the returned gradients: ")
    print(start_grads.shape)
    print("Shape of the step sizes: ")
    print(start_step_sizes.shape)

    # now calculate attributions from the intermediate gradients

    # multiply by the step sizes
    scaled_grads = start_grads.view(n_steps, -1) * start_step_sizes
    # reshape and sum along the num_steps dimension
    scaled_grads = torch.sum(scaled_grads.reshape((n_steps, 1) + start_grads.shape[1:]), dim=0)
    # pass forward the input and baseline ids for reference
    forward_input_ids = model.bert.embeddings.forward(input_ids)
    forward_baseline_ids = model.bert.embeddings.forward(baseline_ids)
    # multiply the scaled gradients by the difference of inputs and baselines to obtain attributions
    attributions = scaled_grads * (forward_input_ids - forward_baseline_ids)
    print("Attributions calculated from intermediate gradients: ")
    print(attributions.shape)
    print(attributions)

    # compare to layer integrated gradients
    layer_integrated = LayerIntegratedGradients(sequence_forward_func, model.bert.embeddings)
    attrs_start = layer_integrated.attribute(inputs=input_ids,
                                             baselines=baseline_ids,
                                             additional_forward_args=(model, token_type_ids, attention_mask),
                                             n_steps=n_steps,
                                             return_convergence_delta=False)
    print("Attributions from layer integrated gradients: ")
    print(attrs_start.shape)
    print(attrs_start)
    

if __name__ == "__main__":
    main()
