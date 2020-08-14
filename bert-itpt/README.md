Scripts for obtaining the results of sentiment analysis with finetuned BERT-ITPT models.  The model and its configuration are stored in bert_model.py. To follow these tutorials, it is recommended to have at least transformers>=3.0.0.

## Download the model checkpoints
### BERT-ITPT-imdb

Eval accuracy: 95.276
[The IMDB dataset that was used can be found here](https://drive.google.com/drive/folders/1CUBHa8Ct_G13bTcKlMiKg2cRNnuBECs5) and the `train_bert_imdb.py` script was used for finetuning.

### BERT-ITPT-sst

Eval accuracy: 92.09
[The SST dataset can be found here](https://nlp.stanford.edu/sentiment/) and the `train_bert_sst2.py` script was used for finetuning.

### BERT-ITPT-yelp

Eval accuracy: 94.75
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

### commented_classifier_single_layer.py
This is the training script for the IMDB dataset.  It works slightly differently by creating InputExample's for each of the IMDB questions that are then passed through the model.
The data is from [This Google Drive folder](https://drive.google.com/drive/folders/1CUBHa8Ct_G13bTcKlMiKg2cRNnuBECs5) which could be slightly different than the IMDB checkpoint.

### bert_model.py
This file contains the pytorch model setup for the BertForSequenceClassificationModel.
Create the configuration and load the model with the following commands:

       >>> config = BertConfig(vocab_size=30522, type_vocab_size=2)
       >>> model = BertForSequenceClassification(config, 2, [11])
       >>> model_states = torch.load(model_path, map_location=device)
       >>> model.load_state_dict(model_states)

Note that the HuggingFace BertTokenizer that is pretrained at 'bert-large-uncased' can be used to tokenize any inputs to this model.