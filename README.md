This is a Transformer model inspired by the paper "Attention is all you need", which is trained for the task of Neural Machine Translation. The dataset I have used is Multi30k and the task is to translate german to english. 

This code uses Pytorch for the model architecture, and tensorboard to plot the graph and losses.

Make sure to download spacy models "en_core_web_sm" and "de_core_news_sm", using the following commands:

"python -m spacy download en_core_web_sm --quiet"

"python -m spacy download de_core_news_sm --quiet"

Install all the dependencies first.
