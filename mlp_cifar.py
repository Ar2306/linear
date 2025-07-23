from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#categorical encoding of labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#architecture 
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history= model.fit(X_train, y_train, epochs=10, batch_size=64,validation_split=0.2)

# Evaluate the model
model.evaluate(X_train, y_train)


#visualize 
plt.plot(history.history['val_accuracy'],)
plt.show()