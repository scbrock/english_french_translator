# English to French Translator Assistant

## Description
This project is a translator assistant that, through terminal, translates English sentences to French sentences.

## Goal
The goal of this project was to investigate different natural language processing (NLP) models for translating English to French. Models I investigated include: RNN with word index encoding, RNN with one-hot encoding, encoder-decoder, and bidirectional. The model analyses can be found in `en_fr.py`.

## Results
The final model file is `translate.py`. I was able to achieve an 80% sentences translation accuracy and a 97% word translation accuracy (from English to French).

## Acknowledgements
The data I used was from [Susan Li](https://towardsdatascience.com/neural-machine-translation-with-python-c2f0a34f7dd) and the code I developed was motivated from Susan Li's code as well. I used Susan's code as starter code and made changes to get my final model and build the terminal interface.

## How To Run:
Please see the `requirements.txt` file for the python requirements. The code to run requires a GPU and can be called with `test_job.sh`.

