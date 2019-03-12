# Day_02_05_rnn_3.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


# 전처리 직접 구현
def make_onehot_1(word):
    # unique함을 보장하기 위해 list가 아닌 set을 사용
    idx2char = sorted(set(word))
    char2idx = {c : i for i, c in enumerate(idx2char)}

    print(idx2char)     # ['e', 'n', 'o', 'r', 's', 't']
    print(char2idx)     # {'e': 0, 'n': 1, 'o': 2, 'r': 3, 's': 4, 't': 5}
    print()

    word_idx = [char2idx[c] for c in word]
    print(word_idx)     # [5, 0, 1, 4, 2, 3]

    x = word_idx[:-1]   # [5, 0, 1, 4, 2]
    y = word_idx[1:]    # [0, 1, 4, 2, 3]

    eye = np.identity(len(char2idx), dtype=np.float32)
    print(eye)

    x_onehot = eye[x]
    x_onehot = x_onehot[np.newaxis]
    # x_onehot = x_onehot.reshape(-1, x_onehot.shape[0], x_onehot.shape[1])
    print(x_onehot)
    print(x_onehot.shape)   # (1, 5, 6)

    y = np.reshape(y, newshape=[-1, len(y)])
    print(y.shape)

    return x_onehot, y, np.array(idx2char)

# 쉬운 방법으로 구현
def make_onehot_2(word):
    #list로 바꿔주는 코드가 꼭 필요하다
    data = list(word)

    lb = preprocessing.LabelBinarizer()
    lb.fit(data)

    onehot = lb.transform(data)
    print(onehot)
    print(lb.fit_transform(data))

    x = np.float32(onehot[:-1])
    y = np.argmax(onehot[1:], axis=1)

    # 차원 추가
    x = x[np.newaxis]
    y = y[np.newaxis]

    print(x.shape, y.shape)
    print(lb.classes_)          # ['e' 'n' 'o' 'r' 's' 't']

    return x, y, lb.classes_

def rnn_3(word, n_iterations=100):
    # x, y, vocab = make_onehot_1(word)
    x, y, vocab = make_onehot_2(word)
    # return

    # vocab = np.array(['e','n','o','r','s','t'])
    # x는 3차원
    # x = [[[0, 0, 0, 0, 0, 1],    # t
    #       [1, 0, 0, 0, 0, 0],    # e
    #       [0, 1, 0, 0, 0, 0],    # n
    #       [0, 0, 0, 0, 1, 0],    # s
    #       [0, 0, 1, 0, 0, 0]]]   # o

    # y = [[1, 0, 0, 0, 0, 0],    # e
    #      [0, 1, 0, 0, 0, 0],    # n
    #      [0, 0, 0, 0, 1, 0],    # s
    #      [0, 0, 1, 0, 0, 0],    # o
    #      [0, 0, 0, 1, 0, 0]]    # r

    # y = [[0, 1, 4, 2, 3]]           # ensor

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32


    # 수업시간에는 BasicRNNCell을 사용하지만
    # 실제로 구현할 때는 GRUCell, LSTMCell 등을 써서 성능을 최적화할 것
    # 내부로직을 알아야 할 필요가 있는가?
    # 알면 판단 시에 도움은 될 수 있으나, 가성비가 많이 떨어진다(직접 두 개 다 해보는게 빠름)

    # 앞 쪽 레이어
    hidden_size = 7

    #단어의 개수, 단어의 길이, 유니크한 글자의 개수
    batch_size, sequence_len, n_classes = x.shape

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

    # 뒤 쪽 레이어
    w = tf.Variable(tf.random_uniform([batch_size, hidden_size, n_classes], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([n_classes], dtype=tf.float32))

    # z = tf.contrib.layers.fully_connected(inputs=outputs,
    #                                       num_outputs=n_classes,
    #                                       weights_initializer=tf.random_normal_initializer())

    z = tf.layers.dense(inputs=outputs, units=n_classes, activation=None,
                        kernel_initializer=tf.glorot_normal_initializer())      # xavier 초기화

    # ---------------------------------------------------------- #

    loss = tf.contrib.seq2seq.sequence_loss(targets=tf.constant(y),
                                            logits=z,
                                            weights=tf.ones([batch_size, sequence_len]))

    # ---------------------------------------------------------- #

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iterations):
        sess.run(train)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2) # 여기도 수정!!
        # preds = preds[0]
        # preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            print(i, sess.run(loss), preds_arg, ''.join(vocab[preds_arg][0]))
            # print(i, sess.run(loss))

    sess.close()

rnn_3('tensor', 1000)
# rnn_3('coffee')
# rnn_3('deep learning')
# rnn_3('Hello, my name is mincheol Shin.', 1000)