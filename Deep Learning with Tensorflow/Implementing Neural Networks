#Implementing neural networks for  WHO life expectancy data

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv('life_expectancy.csv')

#observing the dataset
print(dataset.head())
print(dataset.describe())

#label for section
dataset = dataset.drop(['Country'], axis = 1)

labels = dataset.iloc[:,-1]
features = dataset.iloc[:, 0:-1]

features = pd.get_dummies(dataset)

#split data into training set

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.35, random_state = 5)

#standardize numerical features

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)

#building model

my_model = Sequential()

input= InputLayer(input_shape = (features.shape[1],))
my_model.add(input)

my_model.add(Dense(64, activation = "relu"))
my_model.add(Dense(1))

print(my_model.summary())

#initializing the optimizer and compiling the model

opti = Adam(learning_rate = 0.01)

my_model.compile( loss = 'mse', metrics = ['mae'], optimizer = opti)

#fit and evaluate the model

my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size=1, verbose = 1)

res_me, res_mae = my_model.evaluate(features_train_scaled, labels_test, verbose = 0)

print(res_mse, res_mae)




