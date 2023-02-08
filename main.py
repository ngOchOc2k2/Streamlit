import streamlit as st
import numpy as np
import model_digits
import select_algo_digits
from keras.datasets import mnist
import model_news

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Duong dan den file news_data
PATH_DATA = 'C:\\Users\\ngocl\\OneDrive\\Project IT\\Project I\\Document_Classifier\\text-classification-tutorial\\news_categories\\news_categories.txt'
PATH_STOPWORD = "C:\\Users\\ngocl\\OneDrive - Hanoi University of Science and Technology\\Ml\\Stopword\vietnamese-stopwords-master\\vietnamese-stopwords-dash.txt"

datasets_name = ('Mnist Dataset', 'News Classifier', 'None')
data = st.sidebar.selectbox('Dataset', datasets_name)
col = st.columns(4)

# Digits_data
(X_trains, y_trains), (X_tests, y_tests) = mnist.load_data()
X_train, y_train, X_test, y_test = model_digits.Preprocess(X_test=X_tests, X_train=X_trains, y_test=y_tests, y_train=y_trains)

# News_data
count = {}
for line in open(PATH_DATA, encoding='utf-8', errors='ignore'):
    key = line.split()[0]
    key = key.replace('__label__', '')
    key = key.replace('_', ' ')
    count[key] = count.get(key, 0) + 1



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

        model_digits.plots(X_train=X_trains, y_train=y_train, X_test=X_tests, y_test=y_test)

        algothrim = st.sidebar.selectbox('Select Model', ('None',
                                                          'K-Nearest Neighbour',
                                                          'MLP',
                                                          'K-Means',
                                                          'Softmax Regression',
                                                          'Support Vector Machine',
                                                          'Na'))
        select_algo_digits.Select_Algo(algo=algothrim)
        
    elif data == 'News Classifier':
        model_news.plot_news_classifier(key=key, count_keys=count)
    
    return True


if __name__ == '__main__':
    display(data=data)