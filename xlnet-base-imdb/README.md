Scripts for obtaining the results of sentiment analysis with the finetuned XLNet Base model.  The model and its configuration can be found on [HuggingFace](https://huggingface.co/transformers/model_doc/xlnet.html).  To follow these tutorials, it is recommended to have at least transformers>=3.0.0.

## Download the dataset
[Google Drive with trained XLNet model, ](https://drive.google.com/file/d/1mTfHTA5kdlQLFRtLGZPRTLHTB8nNL-du/view?usp=sharing)

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

### train_xlnet.py
This is the training script for the IMDB dataset.  It works by creating InputExample's for each of the IMDB questions that are then passed through the model.
The data is from [This Google Drive folder](https://drive.google.com/drive/folders/1CUBHa8Ct_G13bTcKlMiKg2cRNnuBECs5) which is preprocessed for this task, removing unneeded characters such as "<br />" and thus is slightly different than the official IMDB checkpoint.

### Loading the pretrained model

       >>> model = XLNetForSequenceClassification.from_pretrained()
       >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       >>> model_states = torch.load("/path/to/model.pth", map_location=device)
       >>> model.load_state_dict(model_states)

Note that the HuggingFace BertTransformer that is pretrained at 'xlnet-base-cased' can be used to tokenize any inputs to this model.