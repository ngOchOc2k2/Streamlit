from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_accuracy_digits(x, y, name_x, name_y, title):
    plt.title(title)
    plt.plot(x, '-o')
    plt.plot(y, '-o')
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend([name_x, name_y])
    st.pyplot()
    return 0


def plot_accuracy_news():
    plt.title('Accuracy News Classifications')
    accuracy_news = {
        'SVM': 82.5,
        'NaiveBayes': 81.6,
        'Linear Regression': 84.13
    }
    
    plot_news = pd.DataFrame.from_dict(accuracy_news, orient='index')
    st.bar_chart(plot_news)