import numpy as np
from scipy.optimize import differential_evolution
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

# Загрузка данных
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование изображений в одномерный вектор
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

# Нормализация данных
X_train /= 255
X_test /= 255

# Преобразование меток классов в бинарный формат
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Определение архитектуры нейронной сети
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Определение функции ошибки и оптимизатора
loss_func = 'categorical_crossentropy'
optimizer = Adam()

# Определение функции для вычисления значения функции ошибки
def calculate_loss(w):
    # Присваивание новых значений весов
    model.set_weights(w)
    # Вычисление значения функции ошибки на обучающем наборе данных
    loss = model.evaluate(X_train, Y_train, verbose=0)
    return loss

# Задание границ диапазона значений весов
bounds = [(-1, 1)] * model.count_params()

# Обучение нейронной сети
result = differential_evolution(calculate_loss, bounds, strategy='best1bin', maxiter=10, popsize=10)

# Присваивание найденных значений весов
model.set_weights(result.x)

# Оценка точности модели на тестовом наборе данных
accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (accuracy[1]*100))
