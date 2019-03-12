# Day_02_04_nltk_gender.py
import nltk
import random

def get_names():
    # corpus : 말뭉치
    # 영화 댓글, 감정 분석 등
    # nltk.download('names')
    print(nltk.corpus.names.fileids()) # 사람 이름에 대한 말뭉치 ['female.txt', 'male.txt']

    # print(nltk.corpus.names.raw('female.txt'))      # 원본 그대로
    # print(nltk.corpus.names.words('female.txt'))    # split해서 단어별로

    females = [(name, 'female') for name in nltk.corpus.names.words('female.txt')]
    males = [(name, 'male') for name in nltk.corpus.names.words('male.txt')]

    print(females[:2])
    print(males[:2])

    names = females + males
    random.shuffle(names)

    print(names[:5])

    return names
    # nltk가 딥러닝이 원하는 모습과 달라서 조금 불편하지만, 전처리가 필요하다

def make_train_test(names, gender_features):
    data = [(gender_features(name), gender) for name, gender in names]
    return data[:6000], data[6000:]


def gender_identification_1(names):
    def gender_features(word):
        return {'last_letter': word[-1]}

    train_set, test_set = make_train_test(names, gender_features)
    print(*train_set[:3], sep='\n')

    # NaiveBayesClassifier : 조건부 확률을 이용
    clf = nltk.NaiveBayesClassifier.train(train_set)

    print(clf.classify(gender_features('Neo')))
    print(clf.classify(gender_features('Trinity')))

    acc = nltk.classify.accuracy(clf, test_set)
    print(acc)

    # 어떤 데이터를 틀리는지에 대한 확인 작업이 필수적이다!! 그래야 성장이 있다!!
    clf.show_most_informative_features(5)

    return acc

def gender_identification_2(names):
    def gender_features(word):
        features = {}
        features['first_letter'] = word[0].lower()
        features['last_letter'] = word[-1].lower()
        
        # features를 과하게 만드는 사례를 보여줌(52개의 feature를 여기서 만듦)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count({})'.format(letter)] = word.lower().count(letter)
            features['has({})'.format(letter)] = (letter in word.lower())

        return features

    train_set, test_set = make_train_test(names, gender_features)
    print(train_set[:3])

    # NaiveBayesClassifier : 조건부 확률을 이용
    clf = nltk.NaiveBayesClassifier.train(train_set)

    return nltk.classify.accuracy(clf, test_set)

def gender_identification_3(names):
    def gender_features(word):
        return {'suffix_1': word[-1], 'suffix_2': word[-2]}

    train_set, test_set = make_train_test(names, gender_features)
    print(train_set[:3])

    # NaiveBayesClassifier : 조건부 확률을 이용
    clf = nltk.NaiveBayesClassifier.train(train_set)

    return nltk.classify.accuracy(clf, test_set)

def gender_identification_4(names):
    def gender_features(word):
        return {'suffix_1': word[-1], 'suffix_2': word[-2:]}

    train_set, test_set = make_train_test(names, gender_features)
    print(train_set[:3])

    # NaiveBayesClassifier : 조건부 확률을 이용
    clf = nltk.NaiveBayesClassifier.train(train_set)

    return nltk.classify.accuracy(clf, test_set)


# get_names()를 빼놓는 이유는 같은 순서의 데이터로 확인하기 위해
names = get_names()     # 7944

acc_1 = gender_identification_1(names)
acc_2 = gender_identification_2(names)
acc_3 = gender_identification_3(names)
acc_4 = gender_identification_4(names)


print('acc 1 : ', acc_1)
print('acc 2 : ', acc_2)
print('acc 3 : ', acc_3)
print('acc 4 : ', acc_4)