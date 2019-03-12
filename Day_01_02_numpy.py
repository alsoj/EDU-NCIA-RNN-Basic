# Day_01_02_numpy.py
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))     # exp 같은 함수를 universal function이라 한다

# 지수 함수의 최소값은 0, 최대값은 무한대
# sigmoid function의 최소값은 0, 최대값은 1

a = np.arange(-3, 6)
print(a)

# print(a + 1) # broadcast 연산
# print(a + a) # vector 연산


# print(sigmoid(a))
# print(sigmoid(a) > 0.5)
# print(sigmoid(a) < 0.5)

# binary classification 에서는 아래와 같이 표현 가능
# 0.85 ==> [0.85, 0.15]

# 하지만 여러 개의 결과 값이 있는 경우에는 sigmoid를 사용할 수 없음
# [0.40, 0.35, 0.25]
# [0.10, 0.85, 0.05]        # 0~1 사이, 합계 1

# 전체 합계
print(np.sum(np.exp(a)))    # x^0 = 1

# softmax
def softmax(z):
    s = np.sum(np.exp(z))

    return np.exp(z)/s

preds = softmax(a)


# softmax 또한 하나의 원소의 입장에서 볼 때는 sigmoid가 적용된 것
# 그래서 각각의 원소에 대해 encoding을 해서 변환이 필요하다
# 8 ==> 0 0 0 0 0 0 0 1
# 3 ==> 0 0 0 1 0 0 0 0
print(np.max(preds))

#최대값에 해당하는 색인(index) 값을 찾기
print(np.argmax(preds))

onehot = np.zeros(9, dtype=np.int32)
onehot[-1] = 1
print(onehot)

print(np.argmax(preds) == np.argmax(onehot))

# 단순 인코딩 된 것을 onehot encoding 하는 방법!!!
encode = [0, 0, 1, 1, 2, 2]
onehot = np.eye(3, dtype=np.int32)
print(onehot)
print(onehot[encode])
print(np.argmax(onehot[encode], axis=1))

print('-' * 50)

import tensorflow as tf

def show_tf_matmul(s1, s2):
    # np.prod 함수는 내부를 다 곱함
    m1 = np.prod(s1)
    m2 = np.prod(s2)

    print(m1)
    print(m2)

    a = np.arange(m1, dtype=np.int32).reshape(s1)
    b = np.arange(m2, dtype=np.int32).reshape(s2)

    c = tf.matmul(a, b)

    print('{} @ {} = {}'.format(s1, s2, c.shape))

show_tf_matmul([2, 3], [3, 2])

# tensorflow 의 행렬 곱셈은 아래와 같다. (4행 3열 짜리 행렬 곱셈을 두 번 수행하겠다는 뜻)
# [3, 4]에 맞춰 [4, 3]을 세팅해주고, page에 해당하는 2를 앞에 둔다
# numpy와 tensorflow의 행렬 곱셈은 다르다(numpy는 진짜 행렬 곱셈)
show_tf_matmul([2, 3, 4], [2, 4, 3])

print('-' * 50)

e = np.arange(24).reshape(2, 3, 4)
print(e)

print(np.argmax(e, axis=0)) # (?, 3, 4) ?를 안 쓰겠다는 것 => 3 x 4 = 12   # 페이지 중 큰 값
print(np.argmax(e, axis=1)) # (2, ?, 4) ?를 안 쓰겠다는 것 => 2 x 4 = 8    # 열 중 큰 값
print(np.argmax(e, axis=2)) # (2, 3, ?) ?를 안 쓰겠다는 것 => 2 x 3 = 6    # 행 중 큰 값