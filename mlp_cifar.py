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
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)
# Evaluate the model
model.evaluate(X_test, y_test)
# Predict
sample_images = X_test[:5]
sample_labels = y_test[:5]
predictions = model.predict(sample_images)
results = predictions.argmax(axis=1)
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.title(f'Actual: {sample_labels[i].argmax()} \n Predicted: {results[i]}')
    plt.imshow(sample_images[i])
    plt.axis('off')
plt.show()