# Day_01_03_SoftmaxRegression_iris.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, datasets


# scipy, scikit-learn, sklearn

def get_iris_1():
    iris = np.loadtxt('Data/iris.csv',
                      skiprows=1,
                      delimiter=',',
                      dtype=np.float32)

    np.random.shuffle(iris)

    print(iris)
    print(iris.shape)

    x = iris[:, :-3]        # fancy indexing
    y = iris[:, -3:]

    print(x.shape, y.shape)     # (150, 4) (150, 3)

    return x, y

def get_iris_2():
    iris = np.loadtxt('Data/iris.csv',
                      skiprows=1,
                      delimiter=',',
                      dtype=np.float32)

    # np.random.shuffle(iris)

    x = iris[:, :-3]        # fancy indexing
    y = iris[:, -3:]

    print(x.shape, y.shape)     # (150, 4) (150, 3)

    return x, y


def get_iris_3():
    x, y = datasets.load_iris(return_X_y=True)

    return x, np.eye(3, dtype=np.int32)[y]

# 문제
# 데이터셋을 나눠서
# 70%의 데이터로 학습하고, 30%의 데이터에 대해 정확도를 알려주세요
def iris_1():
    x, y = get_iris_1()

    train_size = int(len(x) * 0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # np.random.shuffle(x)
    # np.random.shuffle(y)
    #
    # print(x)
    # print(y)
    #
    # x_train_set = x[:round(len(x)*0.7)]
    # y_train_set = y[:round(len(x)*0.7)]
    #
    # x_test_set = x[:round(len(x)*0.3)]
    # y_test_set = y[:round(len(x)*0.3)]

    print(x_train.shape, x_test.shape) # (105, 4)(45, 4)
    print(y_train.shape, y_test.shape) # (105, 3)(45, 3)


    x_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([4, 3], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    # z = w[0] * x[:,0] + w[1] + x[:,1] + b
    # z = w[0] * x[:,0] + w[1] + x[:,1] + w[2] + x[:,2]


    # (105, 3) = (105, 4) @ (4, 3)
    z = tf.matmul(x_p, w) + b

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train, feed_dict={x_p: x_train})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x_train}))

    preds_z_train = sess.run(z, {x_p: x_train})
    # print(preds_z_train)

    preds_h_train = sess.run(hx, {x_p: x_train})
    # print(preds_h_train)

    arg_z_train = np.argmax(preds_z_train, axis=1) # 0(수직), 1(수평)
    # print(arg_z_train)

    arg_h_train = np.argmax(preds_h_train, axis=1) # 0(수직), 1(수평)
    # print(arg_h_train)

    arg_y_train = np.argmax(y_train, axis=1)
    # print(arg_y_train)

    equals_train = (arg_z_train == arg_y_train)
    # print(equals_train)
    print('acc_train : ', np.mean(equals_train))

    #################################################################

    preds_z_test = sess.run(z, {x_p: x_test})
    # print(preds_z_test)

    preds_h_test = sess.run(hx, {x_p: x_test})
    # print(preds_h_test)

    arg_z_test = np.argmax(preds_z_test, axis=1) # 0(수직), 1(수평)
    # print(arg_z_test)

    arg_h_test = np.argmax(preds_h_test, axis=1) # 0(수직), 1(수평)
    # print(arg_h_test)

    arg_y_test = np.argmax(y_test, axis=1)
    # print(arg_y_test)

    equals_test = (arg_z_test == arg_y_test)
    # print(equals_test)
    print('acc_test : ', np.mean(equals_test))

    sess.close()

def iris_2():
    x, y = get_iris_2()

    # train_size = int(len(x) * 0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # shuffle = default
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    print(x_train.shape, x_test.shape) # (105, 4)(45, 4)
    print(y_train.shape, y_test.shape) # (105, 3)(45, 3)


    x_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([4, 3], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    # z = w[0] * x[:,0] + w[1] + x[:,1] + b
    # z = w[0] * x[:,0] + w[1] + x[:,1] + w[2] + x[:,2]


    # (105, 3) = (105, 4) @ (4, 3)
    z = tf.matmul(x_p, w) + b

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train, feed_dict={x_p: x_train})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x_train}))

    preds_z_train = sess.run(z, {x_p: x_train})
    # print(preds_z_train)

    preds_h_train = sess.run(hx, {x_p: x_train})
    # print(preds_h_train)

    arg_z_train = np.argmax(preds_z_train, axis=1) # 0(수직), 1(수평)
    # print(arg_z_train)

    arg_h_train = np.argmax(preds_h_train, axis=1) # 0(수직), 1(수평)
    # print(arg_h_train)

    arg_y_train = np.argmax(y_train, axis=1)
    # print(arg_y_train)

    equals_train = (arg_z_train == arg_y_train)
    # print(equals_train)
    print('acc_train : ', np.mean(equals_train))

    #################################################################

    preds_z_test = sess.run(z, {x_p: x_test})
    # print(preds_z_test)

    preds_h_test = sess.run(hx, {x_p: x_test})
    # print(preds_h_test)

    arg_z_test = np.argmax(preds_z_test, axis=1) # 0(수직), 1(수평)
    # print(arg_z_test)

    arg_h_test = np.argmax(preds_h_test, axis=1) # 0(수직), 1(수평)
    # print(arg_h_test)

    arg_y_test = np.argmax(y_test, axis=1)
    # print(arg_y_test)

    equals_test = (arg_z_test == arg_y_test)
    # print(equals_test)
    print('acc_test : ', np.mean(equals_test))

    sess.close()

def iris_3():
    x, y = get_iris_3()

    print(x.shape, y.shape)

    # train_size = int(len(x) * 0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # shuffle = default
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    print(x_train.shape, x_test.shape)  # (105, 4)(45, 4)
    print(y_train.shape, y_test.shape)  # (105, 3)(45, 3)

    x_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([4, 3], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    # z = w[0] * x[:,0] + w[1] + x[:,1] + b
    # z = w[0] * x[:,0] + w[1] + x[:,1] + w[2] + x[:,2]

    # (105, 3) = (105, 4) @ (4, 3)
    z = tf.matmul(x_p, w) + b

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train,
                                                        logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train, feed_dict={x_p: x_train})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x_train}))

    preds_z_train = sess.run(z, {x_p: x_train})
    # print(preds_z_train)

    preds_h_train = sess.run(hx, {x_p: x_train})
    # print(preds_h_train)

    arg_z_train = np.argmax(preds_z_train, axis=1)  # 0(수직), 1(수평)
    # print(arg_z_train)

    arg_h_train = np.argmax(preds_h_train, axis=1)  # 0(수직), 1(수평)
    # print(arg_h_train)

    arg_y_train = np.argmax(y_train, axis=1)
    # print(arg_y_train)

    equals_train = (arg_z_train == arg_y_train)
    # print(equals_train)
    print('acc_train : ', np.mean(equals_train))

    #################################################################

    preds_z_test = sess.run(z, {x_p: x_test})
    # print(preds_z_test)

    preds_h_test = sess.run(hx, {x_p: x_test})
    # print(preds_h_test)

    arg_z_test = np.argmax(preds_z_test, axis=1)  # 0(수직), 1(수평)
    # print(arg_z_test)

    arg_h_test = np.argmax(preds_h_test, axis=1)  # 0(수직), 1(수평)
    # print(arg_h_test)

    arg_y_test = np.argmax(y_test, axis=1)
    # print(arg_y_test)

    equals_test = (arg_z_test == arg_y_test)
    # print(equals_test)
    print('acc_test : ', np.mean(equals_test))

    sess.close()


# softmax_regression_1()
# softmax_regression_2()
# iris_1()
iris_2()
# iris_3()