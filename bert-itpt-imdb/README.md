Scripts for obtaining the results of sentiment analysis with the finetuned BERT model

## Download the dataset
[Google Drive with trained BERT model](https://drive.google.com/file/d/1R_7SVjETSHs74ff2ita7PrahcFbM1gZa/view?usp=sharing)

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