import numpy as np
from keras.models import load_model
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas

st.write('## Nhan dien chu cai viet tay')
col = st.columns(2)

# Load models sau khi train
models = load_model("D:\model_hand.h5")

# Danh sách những chữ cái cần nhận diện
word_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Tạo giao diện nhận diện chữ cái
drawing_mode = 'freedraw'
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)
drawing_mode = 'freedraw'

with col[0]:
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

# Xử lí ảnh và dự đoán
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))
    x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_ori = x_img.reshape(1, 28, 28)
    st.write(x_img.shape)
    st.image(x_img)
    label = models.predict(x_ori)
    with col[1]:
        st.success('#### Accuracy is: ' + word_dict[np.argmax(label)])
