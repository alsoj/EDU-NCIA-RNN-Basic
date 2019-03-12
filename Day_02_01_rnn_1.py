# Day_02_01_rnn_1.py
import tensorflow as tf
import numpy as np

# 'tensor' => enorst
# x : tenso
# y : ensor

# 문제
# tenso를 입력으로 받아서 ensor를 예측하는 모델을 만드세요
def rnn_1_1():
    vocab = np.array(['e','n','o','r','s','t'])

    # e = [1, 0, 0, 0, 0, 0]
    # n = [0, 1, 0, 0, 0, 0]
    # o = [0, 0, 1, 0, 0, 0]
    # r = [0, 0, 0, 1, 0, 0]
    # s = [0, 0, 0, 0, 1, 0]
    # t = [0, 0, 0, 0, 0, 1]

    x = [[0, 0, 0, 0, 0, 1],    # t
         [1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0]]    # o

    y = [[1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0],    # o
         [0, 0, 0, 1, 0, 0]]    # r

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

    w = tf.Variable(tf.random_uniform([6, 6], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([6], dtype=tf.float32))

    # (5, 6) = (5, 6) @ (6, 6)
    z = tf.matmul(x, w) + b

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            print(i, sess.run(loss), preds_arg, vocab[preds_arg])


    arg_text = np.argmax(sess.run(z),axis=1)
    print(arg_text)

    text = np.array(['e', 'n', 'o', 'r', 's', 't'])

    print(text[arg_text])

    sess.close()

def rnn_1_2():
    vocab = np.array(['e','n','o','r','s','t'])

    # e = [1, 0, 0, 0, 0, 0]
    # n = [0, 1, 0, 0, 0, 0]
    # o = [0, 0, 1, 0, 0, 0]
    # r = [0, 0, 0, 1, 0, 0]
    # s = [0, 0, 0, 0, 1, 0]
    # t = [0, 0, 0, 0, 0, 1]

    x = [[0, 0, 0, 0, 0, 1],    # t
         [1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0]]    # o

    y = [[1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0],    # o
         [0, 0, 0, 1, 0, 0]]    # r

    # 2차원 => 3차원 (3차원으로 가공)
    # 그래서 CNN보다 어렵다
    x = np.float32([x])   # Data Type이 맞지 않을 때 최우선 타입은 float32

    # ---------------------------------------------------------- #

    # w = tf.Variable(tf.random_uniform([6, 6], dtype=tf.float32))
    # b = tf.Variable(tf.random_uniform([6], dtype=tf.float32))
    #
    # # (5, 6) = (5, 6) @ (6, 6)
    # z = tf.matmul(x, w) + b
    #
    # # 직접 출력할 일은 거의 없다
    # # 비용계산 시에도 softmax_cross_entropy_with_logits_v2에 softmax가 포함되어 있음
    # # 결과 값을 출력할 때도 argmax를 쓰기 때문에 z만 써도 충분하다
    # hx = tf.nn.softmax(z)

    # 수업시간에는 BasicRNNCell을 사용하지만
    # 실제로 구현할 때는 GRUCell, LSTMCell 등을 써서 성능을 최적화할 것
    # 내부로직을 알아야 할 필요가 있는가?
    # 알면 판단 시에 도움은 될 수 있으나, 가성비가 많이 떨어진다(직접 두 개 다 해보는게 빠름)

    hidden_size = 6

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

    z = outputs[0]

    # ---------------------------------------------------------- #

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train)

        # preds = sess.run(hx)
        # preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            # print(i, sess.run(loss), preds_arg, vocab[preds_arg])
            print(i, sess.run(loss))


    arg_text = np.argmax(sess.run(z),axis=1)
    print(arg_text)

    text = np.array(['e', 'n', 'o', 'r', 's', 't'])

    print(text[arg_text])

    sess.close()

def rnn_1_3():
    vocab = np.array(['e','n','o','r','s','t'])

    # e = [1, 0, 0, 0, 0, 0]
    # n = [0, 1, 0, 0, 0, 0]
    # o = [0, 0, 1, 0, 0, 0]
    # r = [0, 0, 0, 1, 0, 0]
    # s = [0, 0, 0, 0, 1, 0]
    # t = [0, 0, 0, 0, 0, 1]

    x = [[0, 0, 0, 0, 0, 1],    # t
         [1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0]]    # o

    y = [[1, 0, 0, 0, 0, 0],    # e
         [0, 1, 0, 0, 0, 0],    # n
         [0, 0, 0, 0, 1, 0],    # s
         [0, 0, 1, 0, 0, 0],    # o
         [0, 0, 0, 1, 0, 0]]    # r

    # 2차원 => 3차원 (3차원으로 가공)
    # 그래서 CNN보다 어렵다
    x = np.float32([x])   # Data Type이 맞지 않을 때 최우선 타입은 float32

    # ---------------------------------------------------------- #

    # 수업시간에는 BasicRNNCell을 사용하지만
    # 실제로 구현할 때는 GRUCell, LSTMCell 등을 써서 성능을 최적화할 것
    # 내부로직을 알아야 할 필요가 있는가?
    # 알면 판단 시에 도움은 될 수 있으나, 가성비가 많이 떨어진다(직접 두 개 다 해보는게 빠름)

    # 앞 쪽 레이어
    hidden_size = 2

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape, _states.shape) # (1, 5, 3) (1, 3) 5 : Data의 개수, 3 : num_units

    # 뒤 쪽 레이어
    w = tf.Variable(tf.random_uniform([2, 6], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([6], dtype=tf.float32))

    # (5, 6) = (5, 2) @ (2, 6)
    z = tf.matmul(outputs[0], w) + b

    # ---------------------------------------------------------- #

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)

        # preds = sess.run(hx)
        # preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            # print(i, sess.run(loss), preds_arg, vocab[preds_arg])
            print(i, sess.run(loss))


    arg_text = np.argmax(sess.run(z),axis=1)
    print(arg_text)

    text = np.array(['e', 'n', 'o', 'r', 's', 't'])

    print(text[arg_text])

    sess.close()

def rnn_1_4():
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

    # ---------------------------------------------------------- #

    # 수업시간에는 BasicRNNCell을 사용하지만
    # 실제로 구현할 때는 GRUCell, LSTMCell 등을 써서 성능을 최적화할 것
    # 내부로직을 알아야 할 필요가 있는가?
    # 알면 판단 시에 도움은 될 수 있으나, 가성비가 많이 떨어진다(직접 두 개 다 해보는게 빠름)

    # 앞 쪽 레이어
    hidden_size = 2

    #단어의 개수, 단어의 길이, 유니크한 글자의 개수
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

    # sparse & dense
    # sparse array  : 0으로 채워진 데이터에 1로 유의미한 데이터 => 위치만 표시한 것 [0, 1, 4, 2, 3]
    # dense는 반대의 개념
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)

    # loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)

    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)

        # preds = sess.run(hx)
        # preds_arg = np.argmax(preds, axis=1)

        if i % 100 == 0:
            # print(i, sess.run(loss), preds_arg, vocab[preds_arg])
            print(i, sess.run(loss))


    arg_text = np.argmax(sess.run(z),axis=1)
    print(arg_text)

    text = np.array(['e', 'n', 'o', 'r', 's', 't'])

    print(text[arg_text])

    sess.close()

# rnn_1_1()
# rnn_1_2()
# rnn_1_3()
rnn_1_4()