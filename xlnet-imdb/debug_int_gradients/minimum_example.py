from captum.attr import LayerIntegratedGradients
from transformers import XLNetForSequenceClassification, XLNetTokenizer
import torch


model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

sentence = "This is most definitely not a good movie and worse than most of their other moves."                                                                                               
sentence2 = "But this is definitely a fantastic movie and way better than most of their animated movies." 

features = tokenizer([sentence, sentence2], return_tensors='pt', padding=True, truncation=True, max_length=512)

input_ids = features["input_ids"] # Size: [2, 20]
token_type_ids = features["token_type_ids"] # Size: [2, 20]
attention_mask = features["attention_mask"] # Size: [2, 20]
baseline_ids = torch.zeros(input_ids.shape, dtype=torch.int64) # Size [2, 20]

def sequence_forward_func(inputs, model, tok_type_ids, att_mask):
    """Passes forward the inputs and relevant keyword arguments."""
    outputs = model(inputs, token_type_ids=tok_type_ids, attention_mask=att_mask)
    return outputs

lig = LayerIntegratedGradients(sequence_forward_func, model.transformer.word_embedding)

attrs = lig.attribute(inputs=input_ids,
                     baselines=baseline_ids,
                     additional_forward_args=(model, token_type_ids, attention_mask),
                     n_steps=50,
                     target=0,
                     return_convergence_delta=False)
