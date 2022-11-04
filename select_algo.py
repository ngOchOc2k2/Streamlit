import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import main
import model
from streamlit_drawable_canvas import st_canvas
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = main.y_train, main.y_train, main.X_test, main.y_test
X_trains, X_tests = main.X_trains, main.X_tests


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
            weight_para = pd.read_csv('"C:\Users\ngocl\OneDrive\Project IT\Project I\Parameters\softmax.csv"')
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
    elif algo == 'Na':
        col = st.columns(3)
        with col[0]:
            input_shape = st.number_input('Select Input shape')
        with col[1]:
            layer_num = st.number_input('Select Layer in Model')
        with col[2]:
            node_in_layer = st.number_input('Select Node in a layer')
        with col[0]:
            dense_activation = st.selectbox('Select activation function', ('relu', 'softmax', 'linear'))
        with col[1]:
            node_output = st.number_input('Select number node output layer')
        with col[2]:
            function_opt = st.selectbox('Select function optimation', ('Adam', 'RMSprop', 'SDG'))
        with col[0]:
            loss_function = st.selectbox('Select loss function',
                                         ('Categorial Crossentropy', 'Sparse categorical Crossentropy'))
        with col[1]:
            batch_size = st.number_input('Select batch size')
        with col[2]:
            num_epochs = st.number_input('Select number epochs')

        inputs = tf.keras.Input(shape=(input_shape,), name="digits")
        dense_layer_dict = {}

        for i in range(layer_num):
            if i == 0:
                dense_layer_dict[i] = tf.keras.layers.Dense(node_in_layer)(inputs)
            else:
                dense_layer_dict[i] = tf.keras.layers.Dense(node_in_layer, activation=dense_activation)(
                    dense_layer_dict[i - 1])
        outputs = tf.keras.layers.Dense(node_output, name='predictions')(dense_layer_dict[i])
        models = tf.keras.Model(inputs=inputs, outputs=outputs)

        if loss_function == 'Categorial Crossentropy':
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            train_metrics = tf.keras.metrics.CategoricalAccuracy()
            val_metric = tf.keras.metrics.CategoricalAccuracy()
        elif loss_function == 'Sparse categorical Crossentropy':
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
            val_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        if batch_size:
            batch_size = int(batch_size)
            x_train = np.reshape(X_train, (-1, input_shape))
            x_test = np.reshape(X_test, (-1, input_shape))
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
            train_steps_per_epoch = len(x_train)
