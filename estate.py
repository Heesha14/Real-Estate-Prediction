import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


df = pd.read_csv("D:\\SLIIT\\Y3.S2\\Fundamentals of Data Mining - IT3051\\Project\\Flask\\Trial\\real.csv")
df.drop('No', axis=1, inplace=True)

X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

#saving model to disk
pickle.dump(model,open('model.pkl','wb'))

#loading mode to compare the results
model = pickle.load(open('model.pkl','rb'))

X1_transaction_date = 2013.25
X2_house_age = 3.7
X3_distance_to_station = 577.9615
X4_convenience_stores = 6
X5_latitude = 24.97201
X6_longitude = 121.54722

variables = [X1_transaction_date, X2_house_age, X3_distance_to_station, X4_convenience_stores,
                 X5_latitude, X6_longitude]
X_test1 = np.array(variables)
X_test1 = X_test1.reshape((1, -1))
print(model.predict(X_test1))

# real_estate_prediction(X1_transaction_date,X2_house_age,X3_distance_to_nearest_MRT_station,X4_number_of_convenience_stores,X5_latitude,X6_longitude)
