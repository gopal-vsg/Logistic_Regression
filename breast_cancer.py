import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = reg.predict(features)
    return le.inverse_transform(prediction)

data = pd.read_csv('tumor.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=0)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred))

char = ["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

Pres_input_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(len(char)):
    Pres_input_values[i] = int(input(f"Enter the value of {char[i]}: "))

y_input = predict(Pres_input_values)

if y_input == 4:
    print("Predicted class: Cancer Detected")
else:
    print("Predicted class: Cancer Not Detected")