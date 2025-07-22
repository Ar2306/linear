import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Normalize the data

# categorical encoding of labels
# y_train=y_train.astype('float32')/255.0
# y_test=y_test.astype('float32')/255.0

#one hot encoding of labels
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#build the architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(64,activation='relu')) 
model.add(Dense(32,activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile the model

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate the model
model.evaluate(X_test, y_test)

#predict
sample_images= model.predict(X_test[:5])
sample_labels = y_test[:5]
predictions=model.predict(sample_images)
print(predictions)

results = np.argmax(predictions, axis=1)
for i in range(sample_images):
    plt.subplot(1, 5, i+1)
    plt.tile(f'actual label:{y_test[i]} \n predicted label:{results[i]}')
    plt.imshow(sample_images[i])
    