import sklearn.datasets
import numpy as np

breast_cancer=sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.groupby('class').mean()


from sklearn.model_selection import train_test_split

X_train, X_test,Y_train, Y_test= train_test_split(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,stratify=Y, random_state=1)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score

prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy on training data : ",accuracy_on_training_data)

prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on text data : ", accuracy_on_test_data)

input_data = (13.08,15.71,85.63,520,0.1075,0.127,0.04568,0.0311,0.1967,0.06811,0.1852,0.7477,1.383,14.67,0.004097,0.01898,0.01698,0.00649,0.01678,0.002425,14.5,20.49,96.09,630.5,0.1312,0.2776,0.189,0.07283,0.3184,0.08183)
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0:
    print("The breast Cancer is Malignant")
else:
    print("The breast Cancer is Benign")

