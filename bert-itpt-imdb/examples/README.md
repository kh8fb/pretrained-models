## examples
A collection of example files illustrating how to utilize these pretrained models and their intersection with other libraries such as LayerIntegratedGradients or LayerIntermediateGradients

#### intermediate_gradients_example.py
This example uses the BERT-itpt finetuned model with [intermediate gradients](https://github.com/kh8fb/intermediate-gradients).  Running this script prints the intermediate gradients and each step along the way to convert them into finalized integrated gradients. Running the script from the command line requires the path to the Google Drive .pth model states dictionary.