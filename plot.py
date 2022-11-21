from matplotlib import pyplot as plt
import streamlit as st


def plot(x, y, name_x, name_y, title):
    plt.title(title)
    plt.plot(x)
    plt.plot(y)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend([name_x, name_y])
    st.pyplot()
    return 0

