import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plot
from sklearn.metrics import classification_report

# Duong dan den file data
PATH_DATA = 'C:\\Users\\ngocl\\OneDrive\\Project IT\\Project I\\Document_Classifier\\text-classification-tutorial\\news_categories\\news_categories.txt'
PATH_STOPWORD = "C:\\Users\\ngocl\\OneDrive - Hanoi University of Science and Technology\\Ml\\Stopword\vietnamese-stopwords-master\\vietnamese-stopwords-dash.txt"
MODEL_PATH = 'C:\\Users\\ngocl\\OneDrive\\Project IT\\Project I\\Document_Classifier\\text-classification-tutorial\\models'

# Mo file data
with open('C:\\Users\\ngocl\\OneDrive\\Project IT\\Project I\\train.txt', encoding='utf-8', errors='ignore') as fp:
    train_news = fp.readlines()
with open('C:\\Users\\ngocl\\OneDrive\\Project IT\\Project I\\test.txt', encoding='utf-8', errors='ignore') as fp:
    test_news = fp.readlines()

train_label = []
train_text = []
test_label = []
test_text = []

for line in train_news:
    words = line.strip().split()
    train_label.append(words[0])
    train_text.append(' '.join(words[1:]))
    
for line in test_news:
    words = line.strip().split()
    test_label.append(words[0])
    test_text.append(' '.join(words[1:]))


# Encoder Labels
label_encoder = LabelEncoder()
label_encoder.fit(train_label)
y_train = label_encoder.transform(train_label)
y_test = label_encoder.transform(test_label)

def plot_news_classifier(key, count_keys):
    for items in key:
        items = items.replace('__label__', '')
    
    count_keys = pd.DataFrame.from_dict(count_keys, orient='index')
    st.bar_chart(count_keys)
    
    col = st.columns(3)
    with col[0]:
        tempplate_new = st.number_input('Select New', step=1)

    with col[1]:
        select_algo = st.selectbox('Select Algorithm', ('None', 'Naive Bayes', 'SVM', 'Linear Classifier'))
        
    with col[2]:
        check_plot = st.selectbox('Plot Accuracy', ('None', 'Yes'))
        
    if tempplate_new != 0:
        st.write(test_text[tempplate_new])
        st.success('Label is:' + test_label[tempplate_new])
        
    if select_algo == 'Naive Bayes':
        model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
        y_pred = model.predict(test_text)
        st.success('Naive Bayes, Accuracy = ' +  str(np.mean(y_pred == y_test)*100) + '%')
        # pd_class = pd.DataFrame(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))
        # st.table(pd_class)

    elif select_algo == 'Linear Classifier':
        model = pickle.load(open(os.path.join(MODEL_PATH,"linear_classifier.pkl"), 'rb'))
        y_pred = model.predict(test_text)
        st.success('Linear Classifier, Accuracy = ' +  str(np.mean(y_pred == y_test)*100) + '%')

    elif select_algo == 'SVM':
        model = pickle.load(open(os.path.join(MODEL_PATH,"svm.pkl"), 'rb'))
        y_pred = model.predict(test_text)
        st.success('SVM, Accuracy = ' +  str(np.mean(y_pred == y_test)*100) + '%')
    
    if check_plot == 'Yes':
        plot.plot_accuracy_news()