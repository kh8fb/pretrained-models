# pretrained-models
A collection of trained state-of-the-art NLP models

# Installation
Create a conda environment with the command 

    conda create --name pretrained-models python=3.8

Then install the requirements

     pip install -r requirements.txt

## Models and Results
The following table has the model, proposed accuracy from the paper, and eval accuracy results from the saved Bert and XLNet models.

|                                                     | YELP Accuracy                      | IMDB Accuracy                           | SST2 Accuracy                            |
|-----------------------------------------------------|------------------------------------|-----------------------------------------|------------------------------------------|
|  [XLNet-Base](https://arxiv.org/pdf/1906.08237.pdf) | Paper: 98.63(Large), Ours: (Base)  | Paper: 96.79(Large), Ours: 95.504(Base) | Paper: 96.8(Large), Ours: 94.44(Base)    |
|  [Bert-itpt](https://arxiv.org/pdf/1905.05583.pdf)  | Paper: 98.19, Ours:                | Paper: 95.79, Ours: 95.276              | Paper: N/A, Ours:                       |
| [XLNet-Large](https://arxiv.org/pdf/1906.08237.pdf) | Paper: 98.63(Large), Ours: (Large) | Paper: 96.79(Large), Ours: (Large)      | Paper: 96.8(Large), Ours: 95.5275(Large) |

## Datasets used in finetuning
Each model is finetuned on all 3 of these datasets.
* The IMDB reviews dataset is a benchmark with 25,000 positive movie reviews and 25,000 negative movie reviews.
* The SST dataset is a collection of over 230,000 sentiment parse trees for reviews from rottentomatoes.com.  We train on a smaller subsets of all the reviews where the length of the review is at least 10.
* The YELP reviews dataset is a sample of over a million samples from the Yelp Dataset Challenge in 2015.  There are 560,000 training samples and 38,000 testing samples including both negative and positive sentiments.


## bert-itpt Folder
BERT model for sentiment analysis finetuned on the IMDB reviews, SST, or Yelp datasets.
In this directory is a script to get the sentiment analysis of a string directly from the command line as well as Google Drive links to the saved model states from finetuning on each of the above datasets.
Alternatively, a function, `get_prediction`, can be called in any other python script to obtain the model's sentiment prediction of any string. This will return a 1 for positive sentiment or a 0 for negative sentiment predicted by the model.

## xlnet-base Folder
XLNet Base model for sentiment analysis finetuned on the IMDB reviews, SST, or Yelp datasets.
In this directory is a script to get the sentiment analysis of a string directly from the command line as well as Google Drive links to the saved model states from finetuning on each of the above datasets.
Alternatively, a function, `get_prediction`, can be called in any other python script to obtain the model's sentiment prediction of any string. This will return a 1 for positive sentiment or a 0 for negative sentiment predicted by the model.
