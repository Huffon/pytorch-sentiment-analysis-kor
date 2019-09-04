## Sentiment Analysis PyTorch implementations
This repo contains various basic sequential models used to **analyze sentiment.**

Base codes are based on this great [sentiment-analysis tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis).

In this project, I specially used **Korean corpus** [NSMC](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus) to apply torchtext into Korean dataset.

And I also used [**soynlp**](https://github.com/lovit/soynlp) library which is used to tokenize Korean sentence. 
It is really nice and easy to use, you should try :)

<br/>

### Overview
- Number of train data: 105,000
- Number of validation data: 45,000
- Number of test data: 50,000
- Number of possible class: 2 (pos / neg)

```
Example:
{'text': '[액션', '이', '없는', '데도', '재미', '있는', '몇안되는', '영화'], 
 'label': 'pos'}
```

<br/>


### Requirements
```
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.3
soynlp==0.0.493
torch==1.0.1
torchtext==0.4.0
```

<br/>

### Models

- In this repository, following models are implemented to analyze sentiment. Other famous models will be updated.
    1. [Vanilla RNN](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/vanilla_rnn.py) 
    2. [Bidirectional LSTM](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/bidirectional_lstm.py)
    3. [CNN (for Sentence Classification)](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/cnn.py)

<br/>

### Usage
- Before training the model, you should train soynlp tokenizer and build vocabulary using following code. 
- By running this code, you will get **tokenizer.pickle**, **text.pickle** and **label.pickle** which are used to train, 
test model and predict user's input sentence

```
python build_pickle.py
```


- For training, run **main.py** with train mode (default option)

```
python main.py --model MODEL_NAME
```

- For testing, run **main.py** with test mode

```
python main.py --model MODEL_NAME --mode test 
```

- For predicting, run **predict.py** with your input sentence. 
- *Don't forget to wrap your input sentence with double quotation mark !*

```
python predict.py --model MODEL_NAME --input "YOUR_INPUT"
```