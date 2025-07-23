
from keras.layers import Dense,Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#import data
(X_train,y_train), (X_test,y_test) = cifar10.load_data()

#Build the architecture
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Train
history=model.fit(X_train,y_train,epochs=30,batch_size=64,validation_split=0.2)

#Evaluate
test_accuracy,loss = model.evaluate(X_test,y_test)
print(f'test_accuracy:{test_accuracy}')
print(f'loss:{loss}')
#Visualization
plt.plot(history.history['accuracy'],color = 'blue',label='train_accuracy')
plt.plot(history.history['val_accuracy'],color = 'red',label='val_accuracy')
plt.legend()
plt.title('Epochs vs Accuracy')
plt.show()