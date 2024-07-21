import random
from keras.optimizers import SGD
from keras.layers import Dense, LSTM, Embedding, Flatten, Input
from keras.models import Sequential, Model
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Inisialisasi WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Membaca data dari file JSON
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("data.json").read()
intents = json.loads(data_file)

# Memproses setiap pola pertanyaan dan kelasnya
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenisasi setiap kata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Tambahkan dokumen ke korpus
        documents.append((w, intent["tag"]))
        # Tambahkan ke daftar kelas
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatisasi kata-kata dan ubah menjadi huruf kecil, lalu hapus duplikat
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Urutkan kelas
classes = sorted(list(set(classes)))
print(len(documents), "dokumen")
print(len(classes), "kelas", classes)
print(len(words), "kata unik yang sudah dilemmatize", words)

# Simpan kata-kata dan kelas-kelas ke dalam file pickle
pickle.dump(words, open("texts.pkl", "wb"))
pickle.dump(classes, open("labels.pkl", "wb"))

# Membuat data latih
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mengacak data latih dan mengonversi menjadi array numpy
random.shuffle(training)
training = np.array(
    training, dtype=object
)  # Menggunakan dtype=object untuk menghindari masalah panjang yang tidak seragam

train_x = np.array([np.array(entry[0]) for entry in training])
train_y = np.array([np.array(entry[1]) for entry in training])
print("Data latih telah dibuat")

# Membuat model LSTM baru
input_shape = len(train_x[0])
vocabulary = len(words)
output_length = len(classes)  # Sesuaikan dengan jumlah kelas

i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)  # Layer Embedding
x = LSTM(10, return_sequences=True)(x)  # Layer Long Short Term Memory
x = Flatten()(x)  # Layer Flatten
x = Dense(output_length, activation="softmax")(x)  # Layer Dense
model = Model(i, x)

# Kompilasi model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Ubah data latih agar sesuai dengan input LSTM (tambahkan dimensi tambahan)
train_x = np.array(train_x)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

# Melatih model
hist = model.fit(train_x, np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("model_lstm.h5", hist)

print("Model LSTM telah dibuat dan disimpan")
