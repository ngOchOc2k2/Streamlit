import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def index(X):
    ii = 0
    maxx = X[0]
    for i in range(len(X)):
        if X[i] >= maxx:
            maxx = X[i]
            ii = i
    return ii


def sigmoid(X):
    return 1 / (np.exp(-X) + 1)


def Preprocess(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train/255.
    X_test = X_test/255.
    return X_train, y_train, X_test, y_test


def plots(X_train, y_train, X_test, y_test):
    col_data = st.columns(2)
    with col_data[0]:
        name_chart = st.selectbox('Select Plot', ('Bar_chart', 'Line_chart'))
    with col_data[1]:
        name_data = st.selectbox('Select Data', ('Data Test', 'Data Train', 'Both'))

    a = []
    b = []
    if name_data == 'Data Train':
        for i in range(10):
            a.append(X_train[y_train == i].shape[0])
    elif name_data == 'Data Test':
        for i in range(10):
            a.append(X_test[y_test == i].shape[0])
    else:
        for i in range(10):
            a.append(X_test[y_test == i].shape[0])
        for i in range(10):
            b.append(X_train[y_train == i].shape[0])

    labels = np.arange(10)
    chart_data = pd.DataFrame(a)

    if name_chart == 'Bar_chart':
        st.bar_chart(data=chart_data)
    elif name_chart == 'Line_chart':
        st.line_chart(data=chart_data)
        # st.line_chart(data=b)

    return True


def KNN(X_train, y_train, X_test, y_test, neighbors=1, p=2):
    model = KNeighborsClassifier(n_neighbors=neighbors, p=p, weights='distance')
    model.fit(X=X_train, y=y_train)
    A = model.predict(X_test)
    return A


def K_means(X_train, y_train, X_test, y_test, n_cluster=2):
    model = KMeans(n_clusters=10)
    model.fit(X_train)
    lala = model.predict(X_test)
    st.write('Accuracy is', 100 * accuracy_score(lala, y_test))
    return True


def Softmax(X_test):
    data = pd.read_csv(r"C:\Users\ngocl\OneDrive\Project IT\Project I\Parameters\softmax.csv")
    data = data.drop(columns=['0'], axis=1)
    abc = data.dot(X_test.T)
    labels = []
    for i in range(abc.shape[1]):
        labels.append(index(abc[i]))
    abc = np.exp(abc) / np.sum(np.exp(abc))
    return labels, data, abc


def SVM(X_train, y_train):
    model = SVC(kernel='linear', C=1e5)
    model.fit(X=X_train, y=y_train)
    return model.coef_, model.intercept_
