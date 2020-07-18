## week2 - sequence2sequence代码

1. 补充`rnn_decoder.py`,`rnn_encoder.py`,`sequence_to_sequence.py`部分代码
2. util/data_util.py中`load_pkl`函数在读取数据的时候出现了memory错误，修正之后有个词不在词向量的字典中；这里读取词向量改成了gensim加载，不在词向量的中词赋值为None
3. 可优化的地方：词向量加载可以使用`annoy`，这样可以提高词向量的加载速度（已经在代码中添加）

## week3 - inference部分代码

1. 补充`test_helper.py`中`greedy_search`部分代码
2. 重新训练模型
   1. 训练环境：GTX1080ti, 12G显存
   2. 训练参数：
      1. batch_size: 64, 使用128显存会溢出
      2. Epoch: 20
      3. embedding_size: 300
      4. input_sequence_len: 200
      5. output_sequence_len: 40
      6. encoder_unit: 256
      7. decoder_unit: 256
3. 模型保存在`ckpt/seq2seq/`,推理使用`ckpt-20`
4. 训练结果保存在`results`路径下
5. TODO
   1. 句子长度的确认，可以使用`均值+2*方差`重新确定一下句子长度
   2. 数据预处理部分，去`停用词`，添加`用户词典`，保证某些词不被切开
   3. 词向量训练，对于`占位符`的处理，可以将添加占位符的文本重新训练词向量，保证占位符在vocab中；
   4. inference使用`beam-search`代替`greedy-search`
   5. inference`结果重复`的偏多，这里需要重点关注一下怎么处理
   6. `并行训练`，需要看一下如何进行配置（同理`文件预处理`部分可以查看计算机配置进行`并行化`处理）
   7. 模型不同类型的保存以及部署
   8. `评价指标`还没有完成