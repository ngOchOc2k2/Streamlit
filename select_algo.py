import time

import cv2
import keras.optimizers
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
import io
import main
import model

# Data after preprocessing
X_train, y_train, X_test, y_test = main.X_train, main.y_train, main.X_test, main.y_test
# Data Origin
X_trains, X_tests = main.X_trains, main.X_tests

y_train_cate = tf.keras.utils.to_categorical(y_train, 10)
y_test_cate = tf.keras.utils.to_categorical(y_test, 10)


def get_model_summary(models):
    stream = io.StringIO()
    models.summary(print_fn=lambda x: stream.write(x + '\n\n   '))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def Select_Algo(algo):
    if algo == 'Neural Network':
        colx = st.columns(3)
        with colx[0]:
            st.sidebar.slider('Select Layer', 1, 5, 1)
        with colx[1]:
            st.sidebar.slider('Select Node in Layer', 1, 10, 1)

    elif algo == 'KNN':
        neighbors = st.sidebar.slider('Select Neighbors', 1, 10, 1)
        norms = st.sidebar.slider('Select Norm ', 1, 4, 1)
        pre = st.button('Predict')
        if pre:
            labels = model.KNN(X_train, y_train, X_test, y_test, neighbors=neighbors, p=norms)
            st.write('Accuracy is ', 100 * accuracy_score(labels, y_test))

    elif algo == 'KMeans':
        model.K_means(X_train, y_train, X_test, y_test, n_cluster=10)

    elif algo == 'Softmax Regression':
        labels_sm, huhi, abc = model.Softmax(X_test=X_test)

        # Giai thich model
        with st.expander('Softmax Regression'):
            st.write('Model ')

        col = st.columns(2)
        with col[0]:
            pre = st.selectbox('Display weight or accuracy', ('None', 'Accuracy', 'Weight parameters'))
        with col[1]:
            selected_image = st.selectbox('Select display labels', ('None', 'Single', '20 Image'))

        if selected_image == '20 Image':
            for i in range(5):
                a = np.random.randint(0, 10000, 5)
                colima = st.columns(5)
                for j in range(5):
                    with colima[j]:
                        st.image(X_trains[a[j]], caption='This number is ' + str(y_train[a[j]]))

        elif selected_image == 'Single':
            with col[1]:
                x = st.slider('Seclect Image', 0, 10000, 1)
            st.image(X_tests[x], caption='This labels is ' + str(labels_sm[x]), width=70)

        if pre == 'Accuracy':
            st.success('Accuracy is: ' + str(100 * accuracy_score(labels_sm, y_test)) + '%')

        elif pre == 'Weight parameters':
            weight_para = pd.read_csv(r"C:\Users\ngocl\OneDrive\Project IT\Project I\Parameters\softmax.csv")
            weight_para = weight_para.drop(columns=['0'], axis=1)
            st.dataframe(weight_para)

        coll = st.sidebar.columns(2)
        with coll[0]:
            if_draw = st.checkbox('Draw image')
        with coll[1]:
            if_maxtrixdigit = st.checkbox('Show Matrix Digit')

        if if_draw:
            col = st.columns(2)
            with col[0]:
                drawing_mode = 'freedraw'
                st.write('### Draw pixel 28x28')
                stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)
                drawing_mode = 'freedraw'
                canvas_result = st_canvas(
                    stroke_width=stroke_width,
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    update_streamlit=True,
                    height=250,
                    width=250,
                    drawing_mode=drawing_mode,
                    key="canvas",
                )

                if canvas_result.image_data is not None:
                    img = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))
                    x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x_ori = x_img
                    x_img = x_img.reshape(-1, 784)
                    labels_sms, huhis, abcs = model.Softmax(X_test=x_img)

            with col[1]:
                st.warning('##### Accuracy is: ' + str(labels_sms[0]))

        if if_maxtrixdigit:
            st.write(x_ori)

    elif algo == 'SVM':
        a, b = model.SVM(X_train=X_train, y_train=y_train)
        st.write(a)
        st.write(b)
    elif algo == 'MLP':
        # Define parameters
        train_metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        val_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        train_steps_per_epoch = len(X_train)
        x_train = X_train
        x_test = X_test
        train_loss_list = [0]
        train_dataset = []

        col = st.columns(3)
        with col[0]:
            input_shape = st.number_input('Select Input shape', step=1, value=784)
        with col[1]:
            layer_num = st.number_input('Select Layer in Model', step=1, value=3)
        with col[2]:
            node_in_layer = st.number_input('Select Node in a layer', step=1, value=512)
        with col[0]:
            dense_activation = st.selectbox('Select activation function', ('relu', 'softmax', 'linear'))
        with col[1]:
            node_output = st.number_input('Select number node output layer', step=1, value=10)
        with col[2]:
            function_opt = st.selectbox('Select function optimation', ('Adam', 'RMSprop', 'SGD'))
        with col[0]:
            loss_function = st.selectbox('Select loss function',
                                         ('Categorial Crossentropy', 'Sparse categorical Crossentropy'))
        with col[1]:
            batch_size = st.number_input('Select batch size', step=1, value=1)
        with col[2]:
            num_epochs = st.number_input('Select number epochs', step=1, value=3)

        if function_opt == 'Adam':
            opt = keras.optimizers.Adam(learning_rate=0.1)
        elif function_opt == 'RMSprop':
            opt = keras.optimizers.RMSprop(learning_rate=0.1)
        elif function_opt == 'SGD':
            opt = keras.optimizers.SGD(learning_rate=0.1)

        if loss_function == 'Categorial Crossentropy':
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            train_metrics = tf.keras.metrics.Accuracy()
            val_metric = tf.keras.metrics.Accuracy()
        elif loss_function == 'Sparse categorical Crossentropy':
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            train_metrics = tf.keras.metrics.Accuracy()
            val_metric = tf.keras.metrics.Accuracy()

        if batch_size:
            batch_size = int(batch_size)
            x_train = np.reshape(X_trains, (-1, input_shape))
            x_test = np.reshape(X_tests, (-1, input_shape))
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_cate))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
            train_steps_per_epoch = len(x_train) // batch_size

        models = keras.models.Sequential()

        for i in range(layer_num):
            if i == 0:
                models.add(tf.keras.layers.Dense(784, input_dim=784))
            elif i < layer_num-1:
                models.add(tf.keras.layers.Dense(512, activation='relu'))
            else:
                models.add(tf.keras.layers.Dense(10, activation='softmax', name='predictions'))

        @tf.function
        def train_step(x_batch, y):
            """
            Tensorflow function to compute gradient, loss and metric defined globally based on given data and model.
            """
            with tf.GradientTape() as tape:
                logits = models(x_batch, training=True)
                loss_values = loss_fn(y, logits)
            grads = tape.gradient(loss_values, models.trainable_weights)
            opt.apply_gradients(zip(grads, models.trainable_weights))
            train_metrics.update_state(y, logits)
            return loss_values

        @tf.function
        def test_step(x_batch, y):
            """
            Tensorflow function to compute predicted loss and metric using sent data from the trained model.
            """
            val_logits = models(x_batch, training=False)
            val_metric.update_state(y, val_logits)
            return loss_fn(y, val_logits)

        coli = st.columns(2)
        with coli[0]:
            btnTrain = st.button("Train")
        with coli[1]:
            btnShowSummary = st.button("Summary Model")

        if btnShowSummary:
            strSum = get_model_summary(models)
            st.write(strSum)

        if btnTrain:
            # models.fit(x_train, y_train_cate)
            st.write("Starting training with {} epochs...".format(num_epochs))
            for epoch in range(num_epochs):
                print("\nStart of epoch %d" % (epoch,))
                st.write("Epoch {}".format(epoch + 1))
                start_time = time.time()
                progress_bar = st.progress(0.0)
                percent_complete = 0
                train_loss_list = []
                epoch_time = 0
                # Creating empty placeholder to update each step result in epoch.
                st_t = st.empty()

                # Iterate over the batches of the dataset.
                for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    start_step = time.time()
                    loss_value = train_step(x_batch_train, y_batch_train)
                    end_step = time.time()
                    epoch_time += (end_step - start_step)
                    train_loss_list.append(float(loss_value))

                    # Log every 200 batches.
                    if step % 200 == 0:
                        print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                        print("Seen so far: %d samples" % ((step + 1) * batch_size))
                        step_acc = float(train_metrics.result())
                        percent_complete = (step / train_steps_per_epoch)
                        progress_bar.progress(percent_complete)
                        st_t.write("Duration : {0:.2f}s, Training acc. : {1:.4f}".format(epoch_time, float(step_acc)))

                progress_bar.progress(1.0)
        # Display metrics at the end of each epoch.
        train_acc = train_metrics.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))  # Reset training metrics at the end of each epoch
        train_metrics.reset_states()  # Find epoch training loss.
        print(train_loss_list)
        train_loss = round((sum(train_loss_list) / len(train_loss_list)), 5)
        val_loss_list = []

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_loss_list.append(float(test_step(x_batch_val, y_batch_val)))  # Find epoch validation loss.
        val_loss = round((sum(val_loss_list) / len(val_loss_list)), 5)
        val_acc = val_metric.result()
        val_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        st.write(
            "Duration : {0:.2f}s, Training acc. : {1:.4f}, Validation acc.:{2:.4f}".format((time.time() - start_time),
                                                                                           float(train_acc),
                                                                                           float(val_acc)))
