# Day_03_01_rnn_5_final.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

def make_sentences(text, sequence_len=20):
    data = list(text)
    print(data) # If you want to build a ship, don't drum up people to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb.classes_)

    onehot = lb.transform(data)
    print(onehot[:3])
    print(onehot.shape)                 # (171, 26) 171 글자, 26개의 문자

    x = np.float32(onehot[:-1])
    y = np.argmax(onehot[1:], axis=1)
    # print('x ::: ', x)
    # print('y ::: ', y)

    print(x.shape, y.shape)             # (170, 26) (170,)

    idx = [(i, i+sequence_len) for i in range(len(data) - sequence_len)]
    print(idx[:3])  # [(0, 20), (1, 21), (2, 22)] 0~20번째 글자, 1~21번째 글자, 2~22번째 글자 ...


    # xx, yy는 list
    xx = [x[n1:n2] for n1, n2 in idx]
    yy = [y[n1:n2] for n1, n2 in idx]

    # xx, yy를 numpy 값으로 치환
    xx = np.float32(xx)
    yy = np.int32(yy)
    print(xx.shape, yy.shape)       # (151, 20, 26) (151, 20)


    return xx, yy, lb.classes_

def rnn_final(text, sequence_len=20, n_iterations=100):
    x, y, vocab = make_sentences(text, sequence_len)

    # 앞 쪽 레이어
    hidden_size = 7

    print('x.shape ::: ', x.shape)

    #단어의 개수, 단어의 길이, 유니크한 글자의 개수
    batch_size, sequence_len, n_classes = x.shape

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)] # basic RNN cell을 여러개 만들어서 전달
    multi = tf.contrib.rnn.MultiRNNCell(cells)      # 다중 Layer
    outputs, _states = tf.nn.dynamic_rnn(multi, x, dtype=tf.float32)
    # print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

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

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iterations):
        sess.run(train)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2) # 여기도 수정!!

        if i % 100 == 0:
            print(i, sess.run(loss))

    preds = sess.run(z)
    print(preds.shape)
    print(preds[:3])

    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg[:3])

    # 문제
    # 예측 결과를 보기 좋게 출력하세요.
    # 여기서 출력되는 것의 ㄱ 자 형태로 출력하면 된다.
    # 1행은 전체 20글자, 2행부터는 마지막 글자만
    for p in preds_arg:
        print(''.join(vocab[p]))

    print('-' * 50)

    result = '*' + ''.join(vocab[preds_arg[0]])

    for p in preds_arg[1:]:
        result += vocab[p[-1]]

    print(text)
    print(result)

    sess.close()



text = ("If you want to build a ship, don't drum up people "
        "to collect wood and don't assign them tasks and work, "
        "but rather teach them to long for the endless immensity of the sea.")

print(text)
rnn_final(text, n_iterations=1000)

# 고민할 거리
# 첫 줄만 다 쓰고, 나머지 버려지는 건 낭비가 아닌가? 앙상블 (대각선으로 해서 평균을 내면?)