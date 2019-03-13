# Day_03_05_seq2seq.py

# 전자제품 콜센터 가정

# 질문
# 주소를 알려주세요
# 전원이 들어오지 않아요
# 부품이 도착하지 않았어요
# 소리가 너무 크게 나요
# 배송기사는 언제 옵니까?
# 왜 설치 안 해줘요?

# 답변
# 서초구 서운동입니다
# 콘센트가 꽂혀있는지 확인해 주세요
# 전원 케이블이 끊어졌나 봤습니까?
# 내일 도착할 겁니다
# 소리를 줄이세요
# 갈지 안갈지 저도 궁금합니다
# 12월에 만나요
# 설치를 신청 했습니까?
# 신청되어 있지 않습니다

import tensorflow as tf
import numpy as np

def make_vocab_and_dict(data):
    eng = sorted({c for w, _ in data for c in w})
    kor = sorted({c for _, w in data for c in w})

    print('eng ::: ', eng)
    print('kor ::: ', kor)

    # S(start), E(end), P(padding)
    vocab = ''.join(eng + kor) + 'SEP'
    print(vocab)

    char2dic = {c: i for i, c in enumerate(vocab)}
    print(char2dic)

    return vocab, char2dic

def make_batch(data, char2dic):
    eye = np.eye(len(char2dic))

    inputs_enc, inputs_dec, targets_dec = [], [], []
    for eng, kor in data:
        # print(eng, 'S' + kor, kor + 'E')
        inputs_enc .append(eye[[char2dic[c] for c in eng]])
        inputs_dec .append(eye[[char2dic[c] for c in 'S' + kor]])
        targets_dec.append([char2dic[c] for c in kor + 'E'])        #onehot label 사용 x

    return np.float32(inputs_enc), np.float32(inputs_dec), np.float32(targets_dec)

def show_seq2seq(data, char2dic, vocab, pred_list):
    inputs_enc, inputs_dec, targets_dec = make_batch(data, char2dic)
    print(inputs_enc.shape, inputs_dec.shape, targets_dec.shape)

    n_classes, hidden_size = len(vocab), 128

    ph_enc_in = tf.placeholder(tf.float32, [None, None, n_classes])
    ph_dec_in = tf.placeholder(tf.float32, [None, None, n_classes])
    ph_target = tf.placeholder(tf.int32, [None, None])

    enc_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size, name='enc_cell')            # name을 지정하지 않으면, 다른 cell과의 구분이 안 되어 충돌
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, ph_enc_in, dtype=tf.float32)  # 원래는 쓰지 않던 _states를 사용하기 위해 enc_states로

    dec_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size, name='dec_cell')            # name을 지정하지 않으면, 다른 cell과의 구분이 안 되어 충돌
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, ph_dec_in, dtype=tf.float32, initial_state=enc_states)  # 원래는 쓰지 않던 _states를 사용하기 위해 enc_states로

    z = tf.layers.dense(outputs, n_classes, activation=None)


    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_target, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feed = {ph_enc_in: inputs_enc,
            ph_dec_in: inputs_dec,
            ph_target: targets_dec}

    for i in range(1000):
        sess.run(train, feed_dict=feed)
        print(i, sess.run(loss , feed_dict=feed))

    # ------------------------------------------------------------- #

    new_data = [(w, 'P' * len(data[0][1])) for w in pred_list]
    print(new_data) # [('blue', 'PP'), ('hero', 'PP')]

    inputs_enc, inputs_dec, targets_dec = make_batch(new_data, char2dic)
    print(inputs_enc.shape, inputs_dec.shape, targets_dec.shape)            # (2, 4, 30) (2, 3, 30) (2, 3)

    feed = {ph_enc_in: inputs_enc,
            ph_dec_in: inputs_dec,
            ph_target: targets_dec}

    preds = sess.run(z, feed_dict=feed)
    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg)

    results = [[vocab[j] for j in i] for i in preds_arg]

    for decoded in results:
        # print(decoded)
        print(''.join(decoded[:-1]), end=' ')


data = [['food', '음식'], ['wood', '나무'],
        ['blue', '파랑'], ['lamp', '전구'],
        ['wind', '바람'], ['hero', '영웅']]

vocab, char2dic = make_vocab_and_dict(data)

show_seq2seq(data, char2dic, vocab, ['blue', 'hero'])