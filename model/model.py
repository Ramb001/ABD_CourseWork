import tensorflow as tf
import numpy as np

from keras import Sequential, layers, utils, losses
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from src.handlers import get_reviews


class Model:
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.sentences_train = []
        self.sentences_test = []
        self.shape = None

    def build(self):
        self.model = Sequential(
            [
                layers.Input(shape=self.shape),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Flatten(),
                layers.Dense(5, activation="softmax"),
            ]
        )
        self.model.compile(
            loss=losses.CategoricalCrossentropy(),
            optimizer="sgd",
            metrics=["accuracy"],
        )

    def training_model(self):
        print("Model is training!\n")

        file = open(
            "/Users/artemstanko/Documents/Education/Python/ABD Course Work/model/training_data.txt",
            "r",
        )
        links_train = [line.rstrip() for line in file]
        data_train = [review for link in links_train for review in get_reviews(link)]

        reviews = [item["review"] for item in data_train]
        labels = self.label_encoder.fit_transform(
            [item["score"] for item in data_train]
        )

        self.tokenizer.fit_on_texts(reviews)
        sequences = self.tokenizer.texts_to_sequences(reviews)
        max_length = max([len(seq) for seq in sequences])

        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        self.shape = (max_length,)
        self.build()

        hot_labels = utils.to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(
            padded_sequences, hot_labels, test_size=0.2
        )

        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=24,
            batch_size=128,
            validation_data=(x_test, y_test),
        )

    def predict(self, reviews):
        for review in reviews:
            sequences = self.tokenizer.texts_to_sequences([review["review"]])
            max_length = max([len(seq) for seq in sequences])

            padded_sequences = pad_sequences(sequences, maxlen=max_length)
            self.shape = (max_length,)
            self.build()

            prediction = self.model.predict(padded_sequences)
            emotion = self.label_encoder.inverse_transform([prediction.argmax()])[0]
            print(
                f"\nОтзыв:\n{review['review'] if review['review'] != '' else '---Текст отзыва отсутствует---'}\nПредсказываемая оценка товара от 1 до 5:\n{emotion}\n"
            )

    def training_graph(self):
        plt.plot(
            self.history.history["accuracy"],
            label="Доля верных ответов на обучающем наборе",
        )
        plt.plot(
            self.history.history["val_accuracy"],
            label="Доля верных ответов на проверочном наборе",
        )
        plt.xlabel("Эпоха обучения")
        plt.ylabel("Доля верных ответов")
        plt.legend()
        plt.show()
