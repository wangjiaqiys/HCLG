from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
from annoy import AnnoyIndex
import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TODO:
BASE_DIR = '/root/share/HCLG/ZN_qiye/week1/HCLG'

def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)

# TODO: 将词向量保存成Index格式
def save_index(w2v):
    wv_index = AnnoyIndex(256)
    i = 0
    for key in w2v.vocab.keys():
        v = w2v[key]
        wv_index.add_item(i, v)
        i += 1
    wv_index.build(10)
    wv_index.save('wv_index_build10.index')

# TODO: 加载测试
def test():
    wv_index = AnnoyIndex(256)
    wv_index.load('/root/share/HCLG/ZN_qiye/week1/HCLG/work_of_word2vec/model/wv_index_build10.index', prefault=False)
    reverse_word_index = {}
    word_index = {}
    with open('/root/share/HCLG/ZN_qiye/week1/HCLG/data/reverse_vocab.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            reverse_word_index[int(line.split()[0])] = line.split()[1]
    with open('/root/share/HCLG/ZN_qiye/week1/HCLG/data/vocab.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_index[line.split()[0]] = int(line.split()[1])
    for item in wv_index.get_nns_by_item(word_index[u'车'], 11):
    print(reverse_word_index[item])
def build(train_x_seg_path, train_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    """
    通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    your code
    w2v = （one line）
    """
    # TODO:
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path), size=256, window=5)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)
    # test
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)
    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)
    # TODO: 使用annoy将词向量保存成索引方式
    save_index(w2v)

if __name__ == '__main__':
    build('{}/data/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/data/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/data/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/data/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/data/sentences.txt'.format(BASE_DIR))

