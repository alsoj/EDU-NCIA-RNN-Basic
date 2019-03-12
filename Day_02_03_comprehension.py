# Day_02_03_comprehension.py
import random

# 컴프리헨션 : 콜렉션을 만드는 한 줄짜리 반복문

for i in range(5):
    print(i, end=' ')

print()

# _ : placeholder - 변수로 사용하지 않겠다는 의미
for _ in range(5):
    print(0, end=' ')

print()

for i in range(5):
    print([i * i], end=' ')

print()

print([i for i in range(5)])        # 리스트 컴프리헨션
print({i for i in range(5)})        # 셋 컴프리헨션
print({i:'a' for i in range(5)})    # 딕셔너리 컴프리헨션

for _ in range(10):
    print(random.randrange(100), end=' ')
print()

# 문제
# 난수 10개가 들어있는 리스트를 만드세요
print([random.randrange(100) for _ in range(10)])        # 난수 리스트 컴프리헨션

# 홀수 난수 10개가 들어있는 리스트를 만드세요
print([random.randrange(1, 100, 2) for _ in range(10)])        # 홀수 난수 리스트 컴프리헨션

# 리스트에 들어있는 홀수만 출력하세요
t = [random.randrange(100) for _ in range(10)]

for i in t:
    if i % 2:
        print(i, end=' ')
print()

# 컴프리헨션은 for 문이 구성될 수 있는 것들로만 가능하다, 재구성
print(sum([i for i in t if i % 2]))


print('/' * 50)


t1 = [random.randrange(100) for _ in range(10)]
t2 = [random.randrange(100) for _ in range(10)]
t3 = [random.randrange(100) for _ in range(10)]

# 2차원 배열
all = [t1, t2, t3]

print([i for i in all])
print([0 for i in all])
print([sum(i) for i in all])
print(sum([sum(i) for i in all]))

# 2차원 리스트를 1차원으로 변환하세요
for i in all:
    for j in i:
        print(j, end=' ')
print()
print([j for i in all for j in i])      #1
print([[j for j in i] for i in all])    #2

# 문제
# 앞의 코드로부터 홀수만 추출하는 코드를 넣으세요
print([j for i in all for j in i if j % 2])
print([[j for j in i if j % 2] for i in all])
print('/' * 50)

print(list('tensor'))
print([c for c in 'tensor'])
print(list('Tensor'.lower()))
print([c.lower() for c in 'Tensor'])

# zip : 여러개의 컬렉션을 묶어주는 역할을 한다(하나씩 가져와서 tuple로 묶어줌)
print([c for c in zip('tensor', range(6))])
print([c for c, i in zip('tensor', range(6))])
print([(c, i) for c, i in zip('tensor', range(6))])

print({c: i for c, i in zip('tensor', range(6))})
print({i: c for c, i in zip('tensor', range(6))})
