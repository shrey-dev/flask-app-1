# Predicting Survival of Passengers of Titanic Dataset using Logistic Regression

# importing libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# importing dataset
train = pd.read_csv(r'C:\Users\Shrey\Downloads\ML Datasets\titanic_train.csv')

# Data preprocessing - dropping null values
train.dropna(inplace=True)

# Features and target
target = 'Survived'
features = ['Pclass', 'Age', 'SibSp', 'Fare']
x = train[features]
y = train[target]

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(x, y)
lr.score(x, y)

# Saving model
pickle.dump(lr, open('model.pkl', 'wb'))