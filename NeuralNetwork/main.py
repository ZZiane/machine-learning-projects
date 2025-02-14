import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork


X = np.array([[9], [20], [19], [25], [24] ,[42],[17] ,[15] , [6] ,  [1] ,  [100],[24] ])
Y = np.array([[0], [1], [1],   [1],  [1] , [1] , [0] ,[0] ,[0] , [0] ,[1] , [0]])

nn = NeuralNetwork(input_size=1, hidden_size=12, output_size=1)

nn.train(X, Y, epochs=800)

res = np.where(nn.predict(np.array([[40], [16]]))>0.5,"Majeur","Mineur")
print(res)


