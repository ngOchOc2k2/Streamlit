import streamlit as st
import numpy as np
import model
import select_algo
from keras.datasets import mnist

# Xoa streamlit goc ben duoi
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)


datasets_name = ('Mnist Dataset', 'Iris data', 'Predict', 'None')
data = st.sidebar.selectbox('Dataset', datasets_name)
col = st.columns(4)
(X_trains, y_trains), (X_tests, y_tests) = mnist.load_data()
X_train, y_train, X_test, y_test = model.Preprocess(X_test=X_tests, X_train=X_trains, y_test=y_tests, y_train=y_trains)


def display(data):
    if data == 'Mnist Dataset':
        selected_image = st.sidebar.selectbox('Select display Image Training Data', ('None', 'Single', '20 Image'))

        if selected_image == '20 Image':
            for i in range(5):
                a = np.random.randint(0, 60000, 5)
                colima = st.columns(5)
                for j in range(5):
                    with colima[j]:
                        st.image(X_trains[a[j]], caption='This number is ' + str(y_train[a[j]]))

        elif selected_image == 'Single':
            x = st.sidebar.slider('Seclect Image', 0, 60000, 1)
            st.image(X_trains[x], caption='This number is ' + str(y_train[x]), width=70)

        model.plots(X_train=X_trains, y_train=y_train, X_test=X_tests, y_test=y_test)

        algothrim = st.sidebar.selectbox('Select Model', ('None',
                                                          'K-Nearest Neighbour',
                                                          'MLP',
                                                          'K-Means',
                                                          'Softmax Regression',
                                                          'Support Vector Machine',
                                                          'Na'))
        select_algo.Select_Algo(algo=algothrim)
    return True


if __name__ == '__main__':
    display(data=data)