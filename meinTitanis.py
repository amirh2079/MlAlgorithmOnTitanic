import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("E:/programming/DataSets/titanic-dataset/titanic_train.csv")
df_test = pd.read_csv("E:/programming/DataSets/titanic-dataset/titanic_test.csv")


df.index = df.passenger_id



cat = {'male':1, 'female':2}
df['sex'] = df['sex'].map(cat)
cat_embarked = {'C':1, 'Q':2, 'S':3}
df['embarked'] = df['embarked'].map(cat_embarked)

y = df.survived

print(df.columns)

cols = ['passenger_id','name','cabin','home.dest', 'sibsp', 'parch', 'boat', 'body', 'ticket']
df.drop(cols, axis = 1, inplace=True)

print(df.columns)
# x = normalize(df.values, norm='l2', axis=1)
df = df.fillna(df.mean())
x = df


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


# model = DecisionTreeClassifier()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# acc_score = accuracy_score(y_test, y_pred)
# # print(df.describe().columns)
# print(acc_score)



# create tree model
model1 = DecisionTreeClassifier(max_depth=2)
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)
acc_score1 = accuracy_score(y_test, y_pred)
# create reg model
model2 = LogisticRegression()
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
acc_score2 = accuracy_score(y_test, y_pred)
# Create SVM model
model3 = SVC()
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
acc_score3 = accuracy_score(y_test, y_pred)


print('--------------')
print('Decision Tree:', acc_score1)
print('Logistic Reg:', acc_score2)
print('SVC :', acc_score3)

print(df.head(10))
