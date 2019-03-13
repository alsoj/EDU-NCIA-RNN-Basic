# Day_03_02_stock.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection

def minmax_scale(data):
    mx = np.max(data, axis=0)
    mn = np.min(data, axis=0)

    return (data - mn) / (mx - mn + 1e-7)   # 다차원 배열 - 다차원 배열


def rnn_stock_1():
    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')

    print(stock.shape)
    print(stock[:3, 0])

    stock = stock[::-1] # python 문법. reverse해서 가져오겠다
    stock = minmax_scale(stock)
    print(stock[:3, 0])

    # stock = stock[::-2] #reverse해서 두 칸씩 건너뛰겠다
    # print(stock[:3, 0])

    sequence_len = 7
    hidden_size = 10
    output_dim = 1      # 마지막에 몇 개를 출력하겠는가? 1개 (종가-close) Many to One

    x = stock
    y = stock[:,-1]
    y = y[:, np.newaxis]
    print(y.shape)

    xx, yy = [], []

    for i in range(len(y) - sequence_len):
        # print(i + sequence_len)           # 731 : 731 행이 마지막 행이 된다
        xx.append(x[i:i+sequence_len])      # 0~6까지, 1~7까지, 2~8까지 ...
        yy.append(y[  i+sequence_len])      # 7번째(다음 날의 종가)

    xx = np.float32(xx)
    yy = np.float32(yy)

    print(xx.shape, yy.shape)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, xx, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :],    # 마지막 row을 가져오겠다
                        units=output_dim,
                        activation=None)

    loss = tf.reduce_mean((z - yy) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()

# 문제
# 학습 70%, 검사 30%로 나눠서 예측하세요
def rnn_stock_2():
    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')

    print(stock.shape)
    print(stock[:3, 0])

    stock = stock[::-1]  # python 문법. reverse해서 가져오겠다
    stock = preprocessing.minmax_scale(stock)
    print(stock[:3, 0])

    # stock = stock[::-2] #reverse해서 두 칸씩 건너뛰겠다
    # print(stock[:3, 0])

    sequence_len = 7
    hidden_size = 10
    output_dim = 1  # 마지막에 몇 개를 출력하겠는가? 1개 (종가-close) Many to One

    x = stock
    y = stock[:, -1]
    y = y[:, np.newaxis]
    print(y.shape)

    xx, yy = [], []

    for i in range(len(y) - sequence_len):
        # print(i + sequence_len)           # 731 : 731 행이 마지막 행이 된다
        xx.append(x[i:i + sequence_len])  # 0~6까지, 1~7까지, 2~8까지 ...
        yy.append(y[i + sequence_len])  # 7번째(다음 날의 종가)

    xx = np.float32(xx)
    yy = np.float32(yy)
    print('xx, yy shape ::: ', xx.shape, yy.shape)

    # ----------------------------------------------------------- #

    # trainset과 testset 구분
    x_train, x_test, y_train, y_test = model_selection.train_test_split(xx, yy, train_size=0.7, shuffle=False)

    n_features = x_train.shape[-1]
    x_holder = tf.placeholder(tf.float32, shape=[None, sequence_len, n_features])

    # x_train = np.float32(x_train)
    # y_train = np.float32(y_train)

    print('x_train, y_train shape ::: ', x_train.shape, y_train.shape)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    # outputs, _states = tf.nn.dynamic_rnn(cell, xx, dtype=tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_holder, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs[:, -1, :],  # 마지막 row을 가져오겠다
                        units=output_dim,
                        activation=None)

    # loss = tf.reduce_mean((z - yy) ** 2) # Regression에서는 cross-entropy 대신 MSE사용
    loss = tf.reduce_mean((z - y_train) ** 2) # Regression에서는 cross-entropy 대신 MSE사용

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        # sess.run(train)
        # print(i, sess.run(loss))
        _, value = sess.run([train, loss], {x_holder:x_train}) # train과 loss를 같이 하면 속도가 조금 더 빠르다

        if i % 100 == 0 :
            print(i, value)

    preds = sess.run(z, {x_holder: x_test})
    print(preds.shape, y_test.shape)        # (218, 1) (218, 1)

    # 2차원 데이터를 flat하게 만들기
    preds = preds.reshape(-1)
    y_test = y_test.reshape(-1)
    print(preds.shape, y_test.shape)        # (218,) (218,)

    sess.close()

    # plt.plot(y_test, 'o')     # 점으로 표시
    plt.plot(y_test, 'r')       # 빨간색으로 표시
    plt.plot(preds, 'g')        # 초록색으로 표시
    plt.show()

# 문제
#
def rnn_stock_3():
    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')

    print(stock.shape)
    print(stock[:3, 0])

    stock = stock[::-1]  # python 문법. reverse해서 가져오겠다
    stock = preprocessing.minmax_scale(stock)
    print(stock[:3, 0])

    # stock = stock[::-2] #reverse해서 두 칸씩 건너뛰겠다
    # print(stock[:3, 0])

    sequence_len = 7
    hidden_size = 10
    output_dim = 1  # 마지막에 몇 개를 출력하겠는가? 1개 (종가-close) Many to One

    x = stock
    y = stock[:, -1]
    y = y[:, np.newaxis]
    print(y.shape)

    xx, yy = [], []

    for i in range(len(y) - sequence_len):
        # print(i + sequence_len)           # 731 : 731 행이 마지막 행이 된다
        xx.append(x[i:i + sequence_len])  # 0~6까지, 1~7까지, 2~8까지 ...
        yy.append(y[i + sequence_len])  # 7번째(다음 날의 종가)

    xx = np.float32(xx)
    yy = np.float32(yy)
    print('xx, yy shape ::: ', xx.shape, yy.shape)


    # trainset과 testset 구분
    x_train, x_test, y_train, y_test = model_selection.train_test_split(xx, yy, train_size=0.7, shuffle=False)

    n_features = x_train.shape[-1]
    # x_holder = tf.placeholder(tf.float32, shape=[None, sequence_len, n_features])

    # ----------------------------------------------------------- #
    # Keras를 활용한 RNN 구현
    # ----------------------------------------------------------- #


    model = tf.keras.Sequential()                               # keras 모델 생성
    model.add(tf.keras.layers.SimpleRNN(units=hidden_size))     # RNN 레이어 추가
    model.add(tf.keras.layers.Dense(units=output_dim))          # Dense 레이어 추가

    # optimizer 및 loss function 정의
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    model.fit(x_train, y_train,                 # 학습을 한다
              epochs=100,
              batch_size=32,
              verbose=0)                        # 출력 X
    print('acc : ', model.evaluate(x_test, y_test, verbose=0))

    preds = model.predict(x_test, batch_size=32, verbose=0)
    print(preds.shape)

    # # plt.plot(y_test, 'o')     # 점으로 표시
    plt.plot(y_test, 'r')       # 빨간색으로 표시
    plt.plot(preds, 'g')        # 초록색으로 표시
    plt.show()

# rnn_stock_1()
# rnn_stock_2()
rnn_stock_3()