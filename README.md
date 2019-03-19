
<h1>Chatbot deep learning project in Keras</h1>

<p>Project consisted of data scraping (Czech language data), preprocessing, modelling and evaluation of results</p>
<p>Trained on GPU Nvidia Tesla v100 (with the usage of iterator training in order to not overload GPU memory)</p>
<h2></h2>

data used: Heureka.cz customer complaints and eshop representatives reactions

used technologies:

Python
Keras
Tensorflow GPU
BeautiulSoup

Used models and concepts:
- Main model: Sequence to sequence (Seq2seq)
- Word embeddings (Word2Vec)
- Compared GRU vs LSTM cells for Encoder
- Attention module
- Dropout
- Bidirectional RNN training
- Beam search decoder


Next steps:
- Trying to evaluate Transformer model and BERT
- Get more data and better quality data
