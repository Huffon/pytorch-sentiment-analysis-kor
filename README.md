## Sentiment Analysis PyTorch implementations
This repo contains various basic sequential models used to analyze sentiment.

Base codes are based on this great [sentiment-analysis tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis).

In this project, I specially used Korean corpus [NSMC](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus) to apply torchtext into Korean dataset.

<br/>

### Requirements
```
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.3
torch==1.0.1
torchtext==0.4.0
```

<br/>

### Usage

- For training, you run main.py with train mode (default mode)

```
python main.py
```

- For testing, you run train.py with test mode

```
python main.py --mode test 
```