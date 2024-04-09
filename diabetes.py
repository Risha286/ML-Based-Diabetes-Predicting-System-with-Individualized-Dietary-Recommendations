import numpy as np
import pandas as pd

import pickle

loaded_model = pickle.load(open('C:/Users/risha/OneDrive/Desktop/Diabetes prediction/trained_model.sav','rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')