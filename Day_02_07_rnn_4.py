# Day_02_07_rnn_4.py
# Day_02_06_word_rnn.py 파일을 그대로 복사
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# words가 여러 개의 문자열일 때 동작하도록 수정하세요
# (1, 5, 6) => (3, 5, 6)
def make_onehot(words):
    data = list(''.join(words))
    print(data)

    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb.classes_)

    xx, yy = [], []

    # 해당 단어 개수만큼 반복
    for word in words:
        onehot = lb.transform(list(word))

        print(onehot)
        # print(lb.fit_transform(data))

        x = np.float32(onehot[:-1])
        y = np.argmax(onehot[1:], axis=1)

        xx.append(x)
        yy.append(list(y))

        # print(xx.shape, yy.shape)

    return np.float32(xx), yy, lb.classes_

def rnn_4(words, n_iterations=100):
    x, y, vocab = make_onehot(words)

    print('x ::: ', x)
    print('y ::: ', y)
    print('vocab ::: ', vocab)

    # x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32


    # 수업시간에는 BasicRNNCell을 사용하지만
    # 실제로 구현할 때는 GRUCell, LSTMCell 등을 써서 성능을 최적화할 것
    # 내부로직을 알아야 할 필요가 있는가?
    # 알면 판단 시에 도움은 될 수 있으나, 가성비가 많이 떨어진다(직접 두 개 다 해보는게 빠름)

    # 앞 쪽 레이어
    hidden_size = 7

    print('x.shape ::: ', x.shape)

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

        if i % 100 == 0:
            print(i, sess.run(loss))
            print(*preds_arg)
            # print(vocab[preds_arg])
            print([''.join(s) for s in vocab[preds_arg]])



    sess.close()


rnn_4(['tensor', 'coffee', 'yellow'], 1000)
# rnn_4('tensor', 1000)