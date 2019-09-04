## Sentiment Analysis PyTorch implementations
This repo contains various basic sequential models used to **analyze sentiment.**

Base codes are based on this great [sentiment-analysis tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis).

In this project, I specially used **Korean corpus** [NSMC](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus) to apply torchtext into Korean dataset.

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

### Usage
- Before training, you should train tokenizer and build vocabulary using following code. 
- By running this code, you will get tokenizer.pickle, text.pickle and label.pickle which are used to train, test model and predict user input sentence

```
python build_pickle.py
```


- For training, you run main.py with train mode (default mode)

```
python main.py --model MODEL_NAME
```

- For testing, you run train.py with test mode

```
python main.py --model MODEL_NAME --mode test 
```

- For predicting, you run predict.py with the input sentence that you want to try. 
- *Don't forget to wrap your input sentence with double quotation mark !*

```
python predict.py --model MODEL_NAME --input "YOUR_INPUT"
```