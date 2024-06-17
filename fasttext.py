import pandas as pd
from sklearn.model_selection import train_test_split
import fasttext

def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list


data = pd.read_excel('reviews_fasttext.xlsx')

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_data[['Отзыв', 'Марка']].to_csv('train_fasttext.txt', header=None, index=None, sep=' ', mode='w')
test_data[['Отзыв', 'Марка']].to_csv('test_fasttext.txt', header=None, index=None, sep=' ', mode='w')

model = fasttext.train_supervised('train_fasttext.txt', epoch=80, lr=0.1, wordNgrams=2, dim=2, minCount=5, minn=3, maxn=5)

model.save_model("model.bin")
result = model.test('test_fasttext.txt')


