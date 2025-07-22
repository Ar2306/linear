

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X=np.linspace(1,10,100)
print(X)

np.shape(X)

Y=3*X+20+np.random.randn(X.shape[0])

model=Sequential()
model.add(Dense(
  1,input_dim=1,activation='linear'
))

from re import VERBOSE
model.compile(optimizer='sgd',loss='mse')

model.fit(X,Y,epochs=50)
pred=model.predict(X)

plt.scatter(X,Y,label="origianl")
plt.plot(X,pred)

plt.plot()