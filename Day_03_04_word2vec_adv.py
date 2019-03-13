# Day_03_04_word2vec_adv.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gensim

def extract(token_count, target, window_size, tokens):
    start = max(target - window_size, 0)
    end = min(target + window_size, token_count - 1) + 1

    return [tokens[i] for i in range(start, end) if i != target] # context만으로 구성해서 return

def make_vocab_and_dict(corpus, stop_words):
    corpus_by_word = [[word for word in sent.split() if word not in stop_words] for sent in corpus]
    print(corpus_by_word)

    print([sent for sent in corpus])
    print([word for word in 'king is a strong man'.split() if word not in stop_words])

    vocab = sorted({word for sent in corpus_by_word for word in sent})
    print(vocab)

    corpus_idx = [[vocab.index(word) for word in sent] for sent in corpus_by_word]
    print(corpus_idx)
    # [[2, 8, 3], [7, 9, 3], [0, 11, 3], [1, 11, 10], [5, 11, 2], [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]

    return vocab, corpus_idx


def build_dataset(corpus_idx, n_classes, window_size, is_skipgram):
    xx, yy = [], []
    for sent in corpus_idx:
        for i, target in enumerate(sent):
            ctx = extract(len(sent), i, window_size, sent)  # i는 target을 가리킴
            print(ctx)

            if is_skipgram:
                for neighbor in ctx:
                    xx.append(target)
                    yy.append(neighbor)
            else:
                xx.append(ctx)
                yy.append(target)

    print('xx ::: ', xx[:3])
    print('yy ::: ', yy[:3])

    return make_onehot(xx, yy, n_classes, is_skipgram)

def make_onehot(xx, yy, n_classes, is_skipgram):
    x = np.zeros([len(xx), n_classes], dtype=np.float32)
    y = np.zeros([len(xx), n_classes], dtype=np.float32)

    for i, (input, label) in enumerate(zip(xx, yy)):
        y[i, label] = 1

        if is_skipgram:
            x[i, input] = 1
        else:
            z = [[int(pos == j) for j in range(n_classes)] for pos in input]
            x[i] = np.mean(z, axis=0)

    print(x[:3])

    return x, y

def show_word2vec(vocab, corpus_idx, window_size, is_skipgram):
    n_classes = len(vocab)
    n_embeds = 2            # feature의 개수

    x, y = build_dataset(corpus_idx, n_classes, window_size, is_skipgram)

    w_hidden = tf.Variable(tf.random_normal([n_classes, n_embeds]))     # w_hidden 안에 들어있는 수들은 unique 하다 (단어 간 유사도를 포함)
    hidden_layer = tf.matmul(x, w_hidden)

    w_output = tf.Variable(tf.random_normal([n_embeds, n_classes]))
    b_output = tf.Variable(tf.random_normal([n_classes]))

    z = tf.matmul(hidden_layer, w_output) + b_output

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100000):
        sess.run(train)

        if i % 100 == 0:
            print(i, sess.run(loss))
    
    # w_hidden을 더 많이 쓴다? w_output 보다? 더 정확해서
    vectors = sess.run(w_hidden)
    print(vectors)

    show_similarity(vectors, vocab, 'skip-gram' if is_skipgram else 'cbow')

    sess.close()


def show_similarity(vectors, vocab, title):
    for word, (x1, x2) in zip(vocab, vectors):
        plt.text(x1, x2, word)

    ax_min = np.min(vectors, axis=0) - 1
    ax_max = np.max(vectors, axis=0) + 1

    plt.xlim(ax_min[0], ax_max[0])
    plt.ylim(ax_min[1], ax_max[1])

    plt.title(title)
    plt.show()

def show_word2vec_from_gensim(corpus, vocab, stop_words, is_skipgram):
    corpus_by_word = [[word for word in sent.split() if word not in stop_words] for sent in corpus]
    print(corpus_by_word)


    vectors = gensim.models.Word2Vec(corpus_by_word,
                                     size=2,            # feature의 개수
                                     window=1,
                                     min_count=1,       # 불용어 (한 번 나온거까지 쓰겠다)
                                     sg=is_skipgram,
                                     alpha=0.1)

    print(vectors)
    print(vectors.wv)
    print(vectors.wv.vectors) # 실제 vector data

    print(vectors['king'])

    show_similarity(vectors.wv.vectors, vocab, 'skip-gram' if is_skipgram else 'cbow')


corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']

vocab, corpus_idx = make_vocab_and_dict(corpus, ['is', 'a', 'will', 'be'])       # 불용어(사용하지 않는 단어들)

# show_word2vec(vocab, corpus_idx, window_size=1, is_skipgram=True)
# show_word2vec(vocab, corpus_idx, window_size=1, is_skipgram=False)

show_word2vec_from_gensim(corpus, vocab, ['is','a','will','be'], is_skipgram=True)



# 아래와 같은 데이터를 Look up table 이라고 부른다.
# [[ 3.3446867  -5.41735   ]
#  [-0.25529054  6.0814857 ]
#  [-2.3454802   0.9420556 ]
#  [-6.4563084   4.403986  ]
#  [ 6.162603    2.6983318 ]
#  [ 1.5209521  -3.4896274 ]
#  [ 4.0974855   2.0590606 ]
#  [-5.9142513  -0.8154775 ]
#  [ 6.84542    -0.5315824 ]
#  [ 4.2173514   5.191948  ]
#  [-2.785399   -7.3651195 ]
#  [ 1.5948781   0.3247859 ]]