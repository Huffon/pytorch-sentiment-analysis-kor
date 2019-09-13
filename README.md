## Sentiment Analysis PyTorch implementations
This repo contains various sequential models used to **classify sentiment of sentence.**

Base codes are based on this great [**sentiment-analysis tutorial**](https://github.com/bentrevett/pytorch-sentiment-analysis).

In this project, I specially used **Korean corpus** [**NSMC**](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus) to apply torchtext into Korean dataset.

And I also used [**soynlp**](https://github.com/lovit/soynlp) library which is used to tokenize Korean sentence. 
It is really nice and easy to use, you should try if you handle Korean sentences :)

<br/>

### Overview
- Number of train data: 105,000
- Number of validation data: 45,000
- Number of test data: 50,000
- Number of possible class: 2 (pos / neg)

```
Example:
{'text': '['액션', '이', '없는', '데도', '재미', '있는', '몇안되는', '영화'], 
 'label': 'pos'}
```

<br/>


### Requirements

- Following libraries are fundamental to this repo. Since I used conda environment `requirements.txt` has much more dependent libraries. 
- If you encounters any dependency problem, just use following command 
    - `pip install -r requirements.txt`

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

- In this repository, following models are implemented to analyze sentiment of input sentence. Other famous classification also models will be updated!
    1. [Vanilla RNN](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/vanilla_rnn.py) 
    2. [Bidirectional LSTM](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/bidirectional_lstm.py)
    3. [CNN (for Sentence Classification)](https://github.com/Huffon/pytorch-sentiment-analysis-kor/blob/master/models/cnn.py)

<br/>

### Usage
- Before training the model, you should train `soynlp tokenizer` on your training dataset and build vocabulary using following code. 
- By running following code, you will get `tokenizer.pickle`, `text.pickle` and `label.pickle` which are used to train, 
test model and predict user's input sentence

```
python build_pickle.py
```


- For training, run `main.py` with train mode (which default option)

```
python main.py --model MODEL_NAME
```

- For testing, run `main.py` with test mode

```
python main.py --model MODEL_NAME --mode test 
```

- For predicting, run `predict.py` with your Korean input sentence. 
- *Don't forget to wrap your input with double quotation mark !*

```
python predict.py --model MODEL_NAME --input "YOUR_INPUT"
```

<br/>

### Example

```
[in]  >> 노잼 뻔한 스토리 뻔한 결말...
[out] >> 0.84 % : Negative
[in]  >> 마음도 따뜻.마요미의 진가. 그리고 감동. 뭐 힐링타임용으로 무난한 가족영화탄생~^^
[out] >> 97.64 % : Positive
[in]  >> 클리쉐 덩어리 예산도 적게들었을듯 한데 마지막 관중조차 CG
[out] >> 26.68 % : Negative
```