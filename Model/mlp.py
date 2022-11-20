import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

@tf.function
def train_step(x, y):
    """
    Tensorflow function to compute gradient, loss and metric defined globally based on given data and model.
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value@tf.function
def test_step(x, y):
    """
    Tensorflow function to compute predicted loss and metric using sent data from the trained model.
    """
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
    return loss_fn(y, val_logits)