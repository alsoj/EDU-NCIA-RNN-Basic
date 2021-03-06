# Day_02_06_word_rnn.py
# Day_02_05_rnn_3.py 파일을 그대로 복사
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

def make_onehot(words):
    data = list(words)

    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(data)

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

def word_rnn(words, n_iterations=100):
    # x, y, vocab = make_onehot_1(word)
    x, y, vocab = make_onehot(words)

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

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
            print(i, sess.run(loss), preds_arg, '*'.join(vocab[preds_arg][0]))
            # print(i, sess.run(loss))

    sess.close()

numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six']
# word_rnn과 char_rnn은 동일한 코드이다. 동일한 logic
# word_rnn(numbers, 1000)

sentence = ' '.join(numbers)
print(sentence)

word_rnn(sentence.split(), 1000)
