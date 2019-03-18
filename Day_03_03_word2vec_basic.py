# Day_03_03_word2vec_basic.py
# word2vec는 단어를 숫자로 변환해주는 것이지만, onehot 인코딩보다 효율이 좋다

def extract(token_count, target, window_size):
    start = max(target - window_size, 0)
    end = min(target + window_size, token_count - 1) + 1

    return [i for i in range(start, end) if i != target] # context만으로 구성해서 return

def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for y in range(token_count):
        x = extract(token_count, y, window_size)
        # print(y, ':', x)

        if is_skipgram:
            print(*[(tokens[y], tokens[i]) for i in x])
        else:
            print([tokens[i] for i in x], tokens[y])

# ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
tokens = 'The quick brown fox jumps over the lazy dog'.split()
print(tokens)

show_dataset(tokens, 2, is_skipgram=True)
# show_dataset(tokens, 2, is_skipgram=False)

#####################################################
# show_dataset(tokens, 2, is_skipgram=True) 출력 결과
#####################################################
# ('The', 'quick') ('The', 'brown')
# ('quick', 'The') ('quick', 'brown') ('quick', 'fox')
# ('brown', 'The') ('brown', 'quick') ('brown', 'fox') ('brown', 'jumps')
# ('fox', 'quick') ('fox', 'brown') ('fox', 'jumps') ('fox', 'over')
# ('jumps', 'brown') ('jumps', 'fox') ('jumps', 'over') ('jumps', 'the')
# ('over', 'fox') ('over', 'jumps') ('over', 'the') ('over', 'lazy')
# ('the', 'jumps') ('the', 'over') ('the', 'lazy') ('the', 'dog')
# ('lazy', 'over') ('lazy', 'the') ('lazy', 'dog')
# ('dog', 'the') ('dog', 'lazy')

#####################################################
# show_dataset(tokens, 2, is_skipgram=False) 출력 결과
#####################################################
# ['quick', 'brown'] The
# ['The', 'brown', 'fox'] quick
# ['The', 'quick', 'fox', 'jumps'] brown
# ['quick', 'brown', 'jumps', 'over'] fox
# ['brown', 'fox', 'over', 'the'] jumps
# ['fox', 'jumps', 'the', 'lazy'] over
# ['jumps', 'over', 'lazy', 'dog'] the
# ['over', 'the', 'dog'] lazy
# ['the', 'lazy'] dog
