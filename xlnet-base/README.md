Scripts for obtaining the results of sentiment analysis with finetuned XLNet Base models.  The model and its configuration can be found on [HuggingFace](https://huggingface.co/transformers/model_doc/xlnet.html).  To follow these tutorials, it is recommended to have at least transformers>=3.0.0.

## Download the model checkpoints
### XLNet-base-imdb
[Google Drive with XLNet Base model trained on IMDB](https://drive.google.com/file/d/1mTfHTA5kdlQLFRtLGZPRTLHTB8nNL-du/view?usp=sharing)

Eval accuracy: 95.504
[The IMDB dataset that was used can be found here](https://drive.google.com/drive/folders/1CUBHa8Ct_G13bTcKlMiKg2cRNnuBECs5) and the `train_xlnet_base_imdb.py` script was used for finetuning.

### XLNet-base-sst
[Google Drive with XLNet Base model trained on SST2](https://drive.google.com/file/d/1K7u0nGi5Ecc8g4A8KEgngDpRs6X5n0Dm/view?usp=sharing)

Eval accuracy: 94.44
[The SST dataset can be found here](https://nlp.stanford.edu/sentiment/) and the `train_xlnet_base_sst2.py` script was used for finetuning.

### XLNet-base-yelp
[Google Drive with XLNet Base model trained on YELP](https://drive.google.com/file/d/1M63fFr4sAhrZ-mNdIJdU8tRUxhzkeQ7j/view?usp=sharing)

Eval accuracy: 98.0
[The `YELP reviews - polarity` dataset can be found here](https://course.fast.ai/datasets/) and the `train_bert_yelp.py` script was used for finetuning.

## Usage

### cli_sentiment.py
This is the command line interface for running sentiment analysis with the pretrained model.
After downloading the model and activating the conda environment, this can be called with

      python3.8 cli_sentiment.py -m /path/to/model.pth -s "I hated this movie\! It was awful\!" --cpu

### get_prediction.py
Script with function, `get_prediction` that can be imported and utilized in other scripts to quickly obtain the setniment analysis of a sentence.
Additionally, you can specify whether the model should run on cuda or cpu.

       >>> from get_prediction import get_prediction
       >>> model_path = "/path/to/model.pth"
       >>> result = get_prediction("I hated this movie! it was awful!!", model_path, cuda=True)

### Loading the pretrained model

       >>> from modified_xlnet import XLNetForSequenceClassification
       >>> model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
       >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       >>> model_states = torch.load("/path/to/model.pth", map_location=device)
       >>> model.load_state_dict(model_states)

Note that the HuggingFace XLNetTokenizer that is pretrained at 'xlnet-base-cased' can be used to tokenize any inputs to this model.

### modified_xlnet.py
This is a slightly modified version of the HuggingFace XLNet implementation. It is necessary in order to run intermediate or integrated gradients to extract attributions from the model.