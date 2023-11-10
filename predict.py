import streamlit as st
from PIL import Image
import joblib
import numpy as np
from PIL import ImageOps

model = joblib.load('Digit_Recognizer.pkl')
st.title("ระบบทำนายตัวเลข")

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัพโหลด', use_column_width=True)

    def predict(image):
        image = ImageOps.grayscale(image)
        # ปรับขนาดรูปภาพเป็น 28x28
        image = image.resize((28, 28))
        # แปลงรูปภาพให้อยู่ในรูปแบบของ NumPy array
        image = np.array(image)
        st.image(image, caption='รูปภาพที่อัพโหลด', use_column_width=True)
        # แปลงให้อยู่ในรูปแบบ 1D array
        image = image.flatten()
        
        st.write(image)
        # ทำนายผลลัพธ์
        prediction = model.predict([image])
        st.write('ผลการทำนาย:', prediction)

    # เพิ่มปุ่มสำหรับทำนาย
    if st.button('ทำนาย'):
        predict(image)
