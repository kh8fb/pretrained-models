# pretrained-models
A collection of trained state-of-the-art NLP models

# Installation
Create a conda environment with the command 

    conda create --name pretrained-models python=3.8
Then install the requirements

     pip install -e requirements.txt

Next, download the bert-itpt dataset from Google Drive [here](https://drive.google.com/file/d/1R_7SVjETSHs74ff2ita7PrahcFbM1gZa/view?usp=sharing)

## bert-itpt-imdb
BERT model for sequence analysis finetuned on the IMDB reviews dataset.
The IMDB reviews dataset is a benchmark with 25,000 positive movie reviews and 25,000 negative movie reviews.
In this directory is a script to get the sentiment analysis of a string directly from the command line.
Alternatively, a function, `get_prediction`, can be called in any other python script to obtain the model's sentiment prediction of any string. This will return a 1 for positive sentiment or a 0 for negative sentiment predicted by the model.