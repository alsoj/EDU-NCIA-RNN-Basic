# Day_01_01_SoftmaxRegression.py
import tensorflow as tf
import numpy as np

#####################
# Predict Grade Basic
#####################
def softmax_regression_1():
    #        feature             target(Label)
    # Study time, Attendace ==> Grades(A, B, C)
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # C
         [9, 8]]

    # Onehot encoding
    y = [[0, 0, 1],     # C
         [0, 0, 1],
         [0, 1, 0],     # B
         [0, 1, 0],
         [1, 0, 0],     # A
         [1, 0, 0]]

    w = tf.Variable(tf.random_uniform([2, 3], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    x = np.float32(x)

    # (6, 3) = (6, 2) @ (2, 3)
    z = tf.matmul(x, w) + b

    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                        logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        # train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=tf.matmul(x, w))))

        if i % 100 == 0:
            print(i, sess.run(loss))

    preds_z = sess.run(z)
    print(preds_z)

    preds_h = sess.run(hx)
    print(preds_h)

    arg_z = np.argmax(preds_z, axis=1)      # 0(vertical), 1(horizontal)
    print(arg_z)

    arg_h = np.argmax(preds_h, axis=1)      # 0(vertical), 1(horizontal)
    print(arg_h)

    arg_y = np.argmax(y, axis=1)
    print(arg_y)

    # Print Accuracy
    equals = (arg_z == arg_y)
    print(equals)
    print('acc : ', np.mean(equals))

    sess.close()


#####################
# Predict Grade - Use Placeholder
#####################
def softmax_regression_2():
    #        feature             target(Label)
    # Study time, Attendace ==> Grades(A, B, C)
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # C
         [9, 8]]

    # Onehot encoding
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = np.float32(x)

    # Add tf.placeholder
    x_p = tf.placeholder(tf.float32)
    y_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([2, 3], dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    z = tf.matmul(x_p, w) + b

    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_p,
                                                        logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x_p: x, y_p: y})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x, y_p: y}))

    preds_h = sess.run(hx, {x_p:[[3, 6], [5, 8]]})      # Test value
    preds_arg = np.argmax(preds_h, axis=1)              # 0(vertical), 1(horizontal)
    print('preds_h ::: ', preds_h)
    print('arg_h ::: ', preds_arg)

    grades = np.array(['A', 'B', 'C'])

    print(grades[preds_arg[0]])
    print(grades[preds_arg[1]])
    print(grades[preds_arg])        # index array

    sess.close()

#####################
# Predict Grade - Use Placeholder
#####################
def softmax_regression_3():
    # feature                     target(Label)
    # 공부한 시간, 출석한 일수 ==> 성적(A, B, C)
    # 1은 dummy feature
    x = [[1, 1, 2],        # C
         [1, 2, 1],
         [1, 4, 5],        # B
         [1, 5, 4],
         [1, 8, 9],        # C
         [1, 9, 8]]

    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

    x_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([3, 3], dtype=tf.float32))
    # b = tf.Variable(tf.random_uniform([3], dtype=tf.float32))

    # z = w[0] * x[:,0] + w[1] + x[:,1] + b
    # z = w[0] * x[:,0] + w[1] + x[:,1] + w[2] + x[:,2]


    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x_p, w)

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                        logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x_p: x})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x}))

    preds_h = sess.run(hx, {x_p:[[1, 3, 6],
                                 [1, 5, 8]]})
    preds_arg = np.argmax(preds_h, axis=1) # 0(수직), 1(수평)
    print('preds_h ::: ', preds_h)
    print('arg_h ::: ', preds_arg)

    grades = np.array(['A', 'B', 'C'])

    print(grades[preds_arg[0]])
    print(grades[preds_arg[1]])
    print(grades[preds_arg])        # index array

    sess.close()

def softmax_regression_4():
    # feature                     target(Label)
    # 공부한 시간, 출석한 일수 ==> 성적(A, B, C)
    # 1은 dummy feature
    x = [[1, 1, 2],        # C
         [1, 2, 1],
         [1, 4, 5],        # B
         [1, 5, 4],
         [1, 8, 9],        # C
         [1, 9, 8]]

    # y = [0, 0, 1, 1, 2, 2]

    # x, y의 data shape을 맞춰주기 위해 경우에 따라서 차원을 변경할 수 있다.
    # y = [[0], [0], [1], [1], [2], [2]]
    # y = [[0, 0, 1, 1, 2, 2]]

    # 문제
    # onehot vector로 y를 만드세요
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    x = np.float32(x)   # Data Type이 맞지 않을 때 최우선 타입은 float32

    x_p = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([3, 3], dtype=tf.float32))

    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x_p, w)

    # 직접 출력할 일은 거의 없다
    hx = tf.nn.softmax(z)

    loss_i = tf.reduce_sum(y * -tf.log(hx), axis=0)

    # loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    # 평균 내는 함수, reduce - 차원을 줄이겠다는 의미(axis를 쓰면 해당 차원을 줄이겠다)
    loss = tf.reduce_mean(loss_i)

    # learning_rate와 같은 hyperparameter 를 잘 정하는 것이 실력이다.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 연산
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x_p: x})

        if i % 100 == 0:
            print(i, sess.run(loss, {x_p: x}))

    preds_h = sess.run(hx, {x_p:[[1, 3, 6],
                                 [1, 5, 8]]})
    preds_arg = np.argmax(preds_h, axis=1) # 0(수직), 1(수평)
    print('preds_h ::: ', preds_h)
    print('arg_h ::: ', preds_arg)

    grades = np.array(['A', 'B', 'C'])

    print(grades[preds_arg[0]])
    print(grades[preds_arg[1]])
    print(grades[preds_arg])        # index array

    sess.close()

# softmax_regression_1()
softmax_regression_2()
# softmax_regression_3()
# softmax_regression_4()
