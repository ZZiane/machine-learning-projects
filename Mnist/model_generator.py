from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import normalize,to_categorical
from keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = normalize(x_train , axis = 1)
x_test = normalize(x_test , axis=1) 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)



model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=512, epochs=10)

model.save('resources/mnist.h5')

