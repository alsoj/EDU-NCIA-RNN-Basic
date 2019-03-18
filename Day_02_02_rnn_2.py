# Day_02_02_rnn_2.py
import tensorflow as tf
import numpy as np

################
# Loss 계산 함수
################
def show_sequence_loss():
    def sequence_loss(targets, logits):
        y = tf.constant(targets)
        z = tf.constant(logits)

        # feature 별로 weights를 다르게 주고 싶을 때 (dummy weight)
        w = tf.ones([1, len(targets[0])])

        loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print(sess.run(loss))

        sess.close()

    # 원래 둘 중 선택하는 거라면 둘 다 쓸 필요가 없지만, 여기서는 예시를 위해서
    # preds1 = [[[0.2, 0.8, 0.0], [0.4, 0.6, 0.0], [0.7, 0.3, 0.0]]]
    preds1 = [[[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]]
    preds2 = [[[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]]

    # x, y에는 차원 1만큼의 차이 존재(우리가 sparse 버전을 쓸 것이기 때문)
    sequence_loss([[1, 1, 1]], preds1)
    sequence_loss([[0, 0, 0]], preds2)

    sequence_loss([[2, 2, 2]], [[[0.2, 0.2, 0.6], [0.4, 0.3, 0.3], [0.2, 0.3, 0.5]]])
    sequence_loss([[1, 1, 1, 1]], [[[0.2, 0.2, 0.6], [0.4, 0.3, 0.3], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]])

###########################
# RNN - sequence_loss 버전 
###########################
def rnn_2_1():
    vocab = np.array(['e','n','o','r','s','t'])

    # e = [1, 0, 0, 0, 0, 0]
    # n = [0, 1, 0, 0, 0, 0]
    # o = [0, 0, 1, 0, 0, 0]
    # r = [0, 0, 0, 1, 0, 0]
    # s = [0, 0, 0, 0, 1, 0]
    # t = [0, 0, 0, 0, 0, 1]

    # x는 3차원
    x = [[[0, 0, 0, 0, 0, 1],    # t
          [1, 0, 0, 0, 0, 0],    # e
          [0, 1, 0, 0, 0, 0],    # n
          [0, 0, 0, 0, 1, 0],    # s
          [0, 0, 1, 0, 0, 0]]]   # o

    # y = [[1, 0, 0, 0, 0, 0],    # e
    #      [0, 1, 0, 0, 0, 0],    # n
    #      [0, 0, 0, 0, 1, 0],    # s
    #      [0, 0, 1, 0, 0, 0],    # o
    #      [0, 0, 0, 1, 0, 0]]    # r

    y = [0, 1, 4, 2, 3]           # ensor
    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

    # 앞 쪽 레이어
    hidden_size = 2

    # batch_size : 단어의 개수
    # sequence_len : 단어의 길이
    # n_classes : 유니크한 글자의 개수
    batch_size, sequence_len, n_classes = x.shape

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

    # 뒤 쪽 레이어
    w = tf.Variable(tf.random_uniform([hidden_size, n_classes], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([n_classes], dtype=tf.float32))

    # (5, 6) = (5, 2) @ (2, 6)
    z = tf.matmul(outputs[0], w) + b

    # ---------------------------------------------------------- #

    y = tf.constant([y])
    z = tf.reshape(z, [batch_size, sequence_len, n_classes])
    # z = tf.reshape(z, [-1, sequence_len, n_classes]) # -1은 계산하기 싫을 때 or 계산하지 못할 때
    # numpy의 reshape은 데이터 가져올 때 / tensor의 reshape은 실행 중 변환해야 할 때

    w_dummy = tf.ones([batch_size, sequence_len])
    loss = tf.contrib.seq2seq.sequence_loss(targets=y, logits=z, weights=w_dummy)

    # ---------------------------------------------------------- #


    # sparse & dense
    # sparse array  : 0으로 채워진 데이터에 1로 유의미한 데이터 => 위치만 표시한 것 [0, 1, 4, 2, 3]
    # dense는 반대의 개념
    # loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)

    # loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)

    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    # loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)

        preds = sess.run(z)
        preds = preds[0]
        preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            print(i, sess.run(loss), preds_arg, vocab[preds_arg])
            # print(i, sess.run(loss))

    arg_text = np.argmax(sess.run(z),axis=1)
    print(arg_text)

    text = np.array(['e', 'n', 'o', 'r', 's', 't'])

    print(text[arg_text])

    sess.close()

###############################
# RNN - sequence_loss 코드 정리
###############################
def rnn_2_2():
    vocab = np.array(['e','n','o','r','s','t'])

    # e = [1, 0, 0, 0, 0, 0]
    # n = [0, 1, 0, 0, 0, 0]
    # o = [0, 0, 1, 0, 0, 0]
    # r = [0, 0, 0, 1, 0, 0]
    # s = [0, 0, 0, 0, 1, 0]
    # t = [0, 0, 0, 0, 0, 1]

    # x는 3차원
    x = [[[0, 0, 0, 0, 0, 1],    # t
          [1, 0, 0, 0, 0, 0],    # e
          [0, 1, 0, 0, 0, 0],    # n
          [0, 0, 0, 0, 1, 0],    # s
          [0, 0, 1, 0, 0, 0]]]   # o

    y = [[0, 1, 4, 2, 3]]           # ensor

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

    # 앞 쪽 레이어
    hidden_size = 2

    #단어의 개수, 단어의 길이, 유니크한 글자의 개수
    batch_size, sequence_len, n_classes = x.shape

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

    # 뒤 쪽 레이어
    w = tf.Variable(tf.random_uniform([batch_size, hidden_size, n_classes], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([n_classes], dtype=tf.float32))

    # (1, 5, 6) = (1, 5, 2) @ (1, 2, 6)
    # z = tf.matmul(outputs, w) + b
    z = tf.contrib.layers.fully_connected(inputs=outputs,
                                          num_outputs=n_classes,
                                          weights_initializer=tf.random_normal_initializer())

    # ---------------------------------------------------------- #

    # y = tf.constant(y)
    # z = tf.reshape(z, [batch_size, sequence_len, n_classes])
    # z = tf.reshape(z, [-1, sequence_len, n_classes]) # -1은 계산하기 싫을 때 or 계산하지 못할 때
    # numpy의 reshape은 데이터 가져올 때 / tensor의 reshape은 실행 중 변환해야 할 때

    # w_dummy = tf.ones([batch_size, sequence_len])
    loss = tf.contrib.seq2seq.sequence_loss(targets=tf.constant(y),
                                            logits=z,
                                            weights=tf.ones([batch_size, sequence_len]))

    # ---------------------------------------------------------- #

    # sparse & dense
    # sparse array  : 0으로 채워진 데이터에 1로 유의미한 데이터 => 위치만 표시한 것 [0, 1, 4, 2, 3]
    # dense는 반대의 개념
    # loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)

    # loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)

    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    # loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)

        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2) # 여기도 수정!!
        # preds = preds[0]
        # preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            print(i, sess.run(loss), preds_arg, vocab[preds_arg])
            # print(i, sess.run(loss))

    sess.close()

# show_sequence_loss()

# rnn_2_1()
rnn_2_2()
