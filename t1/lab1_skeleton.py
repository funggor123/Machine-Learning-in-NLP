# author: ‘CHENG CHI FUNG'
# student_id: '12219691'
import nltk as tk

#tk.download('punkt')

def q1(words):
    print('q1: {:}'.format(words))
    # 1. Print all words beginning with sh
    # YOUR CODE
    for word in words:
        if len(word) > 1:
            if word[0] == 's' and word[1] == 'h':
                print(word, end=" ")
    print("")

    # 2. Print all words longer than four characters
    # List Comprehensions
    # YOUR CODE
    for word in words:
        if len(word) > 4:
            print(word, end=" ")


def q2(file_name):
    print('q2: {:}'.format(file_name))
    # YOUR CODE
    # readfile
    # Your Code
    read_file = open("lab1_text.txt").read()

    words = []
    # process text using regex or nltk.word_tokenize()
    # Your Code
    words = tk.tokenize.word_tokenize(read_file)

    # 1. Find words ending in 'ize'
    print("1. Find words ending in 'ize'")
    # Your Code
    for word in words:
        length = len(word)
        if len(word) > 2:
            if word[length - 1] == 'e' and word[length - 2] == 'z' and word[length - 3] == 'i':
                print(word, end=" ")
    print("")

    # 2. Find words containing 'z'
    print("2. Find words containing 'z'")
    # Your Code
    for word in words:
        for ch in word:
            if ch == 'z':
                print(word, end=" ")
                break
    print("")

    # 3. Containing the sequence of letters "pt"
    print("3. Find words containing 'pt'")
    # Your Code
    for word in words:
        if "pt" in word:
            print(word, end=" ")

    print("")

    # 4. Find words that are in titlecase
    print("4. Find words that are in titlecase")
    for word in words:
        if word.istitle():
            print(word, end=" ")


def q3(line):
    print('q3: {:}'.format(line))
    # YOUR CODE
    words = []
    tokenizer = tk.tokenize.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(line)
    concat_word = ""
    for word in words:
        concat_word += word
    pad = True
    concat_word = concat_word.lower()
    for i in range(len(concat_word) // 2):
        if concat_word[i] != concat_word[len(concat_word) - i - 1]:
            pad = False
    print(pad)


if __name__ == '__main__':
    q1_input = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
    q1(q1_input)

    print()
    q2_input = 'lab1_text.txt'
    q2(q2_input)

    print()
    q3_input = ['A man, a plan, a canal: Panama', 'race a car', 'raca a                              car']
    for q3_in in q3_input:
        q3(q3_in)
