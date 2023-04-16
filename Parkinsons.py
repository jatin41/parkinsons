

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('/Users/jatinagrawal/__pycache__/parkinsons (1).csv')

# printing the first 5 rows of the dataframe
parkinsons_data.head()

# number of rows and columns in the dataframe
parkinsons_data.shape

# getting more information about the dataset
parkinsons_data.info()

# checking for missing values in each column
parkinsons_data.isnull().sum()

# getting some statistical measures about the data
parkinsons_data.describe()

# distribution of target Variable
parkinsons_data['status'].value_counts()



# grouping the data bas3ed on the target variable
parkinsons_data.groupby('status').mean()



X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

print(X)

print(Y)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)



scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

print(X_train)



model = svm.SVC(kernel='linear')

# training the SVM model with training data
model.fit(X_train, Y_train)



# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data : ', test_data_accuracy)


a=st.number_input("MDVP:Fo(Hz)",value=0.0, step=0.000001, format="%.6f")
b=st.number_input("MDVP:Fhi(Hz)",value=0.0, step=0.000001, format="%.6f")
c=st.number_input("MDVP:Flo(Hz)",value=0.0, step=0.000001, format="%.6f")
d=st.number_input("MDVP:Jitter(%)",value=0.0, step=0.000001, format="%.6f")
e=st.number_input("MDVP:Jitter(Abs)",value=0.0, step=0.000001, format="%.6f")
f=st.number_input("MDVP:RAP",value=0.0, step=0.000001, format="%.6f")
g=st.number_input("MDVP:PPQ",value=0.0, step=0.000001, format="%.6f")
h=st.number_input("Jitter:DDP",value=0.0, step=0.000001, format="%.6f")
i=st.number_input("MDVP:Shimmer",value=0.0, step=0.000001, format="%.6f")
j=st.number_input("MDVP:Shimmer(dB)",value=0.0, step=0.000001, format="%.6f")
k=st.number_input("Shimmer:APQ3",value=0.0, step=0.000001, format="%.6f")
l=st.number_input("Shimmer:APQ5",value=0.0, step=0.000001, format="%.6f")
m=st.number_input("MDVP:APQ",value=0.0, step=0.000001, format="%.6f")
n=st.number_input("Shimmer:DDA",value=0.0, step=0.000001, format="%.6f")
o=st.number_input("NHR",value=0.0, step=0.000001, format="%.6f")
p=st.number_input("HNR",value=0.0, step=0.000001, format="%.6f")
q=st.number_input("RPDE",value=0.0, step=0.000001, format="%.6f")
r=st.number_input("DFA",value=0.0, step=0.000001, format="%.6f")
s=st.number_input("spread1",value=0.0, step=0.000001, format="%.6f")
t=st.number_input("spread2",value=0.0, step=0.000001, format="%.6f")
u=st.number_input("D2",value=0.0, step=0.000001, format="%.6f")
v=st.number_input("PPE",value=0.0, step=0.000001, format="%.6f")


input_data = (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if st.button("Predict"):
    if (prediction[0] == 0):
        st.write("The Person does not have Parkinsons Disease")
    else:
        st.write("The Person has Parkinsons")