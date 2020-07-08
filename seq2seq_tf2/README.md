## week2 - sequence2sequence代码

1. 补充`rnn_decoder.py`,`rnn_encoder.py`,`sequence_to_sequence.py`部分代码
2. util/data_util.py中`load_pkl`函数在读取数据的时候出现了memory错误，修正之后有个词不在词向量的字典中；这里读取词向量改成了gensim加载，不在词向量的中词赋值为None
3. 可优化的地方：词向量加载可以使用`annoy`，这样可以提高词向量的加载速度（已经在代码中添加）