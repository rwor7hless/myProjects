import string
import openpyxl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# Загрузка списка слов из файла со словами
def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list
def remove_punctuation(text):
    # Удаляем знаки препинания из текста
    cleaned_text = ''
    for char in text:
        if char in string.punctuation:
            cleaned_text += ' '  # Добавляем пробел вместо знака препинания
        else:
            cleaned_text += char.lower()
    return cleaned_text

def filter_text(text, word_list):
    text = remove_punctuation(text)
    words = text.split()
    filtered_text = []
    negate = False  # флаг, который будет указывать на текущий отрицательный контекст

    for i, word in enumerate(words):  # исправлено: использование enumerate для получения индекса и слова
        if i < len(words) - 1:  # проверка, что индекс не выходит за границы списка
            next_word = words[i+1]
        else:
            next_word = None

        if word == "не" and (root in next_word for root in word_list):  # исправлено: добавлена проверка следующего слова
            negate = True
        elif not any(root in word for root in word_list):
            negate = False
        elif any(root in word for root in word_list):  # проверяем наличие корня из списка слов в текущем слове
            if negate:  # если флаг отрицания установлен
                filtered_text.append("не " + word)
                negate = False
            else:  # если отрицание неактивно, добавляем слово без "не"
                filtered_text.append(word)
        elif "не" in word and word != "не" and word in word_list:  # если слово содержит "не", но не в списке отрицательных слов
            filtered_text.append(word)

    return ' '.join(filtered_text)




def load_data(file_path, word_list):
    workbook = openpyxl.load_workbook(file_path, read_only=True)
    sheet = workbook.active

    texts = []
    labels = []

    for row in sheet.iter_rows(min_row=1, max_row=1000, values_only=True):
        text = row[0] if row[0] is not None else ""
        filtered_text = filter_text(text, word_list)
        texts.append(filtered_text)

        label = row[1] if row[1] is not None else 0
        labels.append(int(label))

    print(texts, labels)
    return texts, labels

word_list_path = 'words.txt'
word_list = load_word_list(word_list_path)

excel_file_path = '../reviews.xlsx'

texts, labels = load_data(excel_file_path, word_list)

filtered_texts = [filter_text(text, word_list) for text in texts]

tokenizer = Tokenizer(num_words=len(word_list), oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(filtered_texts)
padded_sequences = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, shuffle=False)

model = Sequential()
model.add(Embedding(input_dim=len(word_list), output_dim=2, input_length=500))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



# Компиляция модели
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(X_train, np.array(y_train), epochs=50, batch_size=2, validation_split=0.2, callbacks=[early_stopping])
predictions_train = model.predict(X_train)


# Построение графика точности
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Построение графика функции потерь
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


print(len(word_list))


model.save('model_seq.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

