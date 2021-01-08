import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PAGE_CONFIG = {"page_title":"OpenCV Playground","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

@st.cache
def load_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def draw_cnts(original, binary):
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, cnts[0], -1, (120,255,4), 2)
    st.image(result, use_column_width=True)


menu = ['Segmentation', 'Blur']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Segmentation':
    st.subheader('Image Segmentation')
    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'], key=3)
    if image_file is not None:
        file_details = {"Filename":image_file.name,
                        "FileType":image_file.type,
                        "FileSize":image_file.size}
        st.write(file_details)

        img = load_image(image_file)
        st.image(img, caption='Original', use_column_width=True)
        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        col1, col2 = st.beta_columns(2)

        with col1:
            if st.checkbox('Blur it!'):
                blur_option = st.selectbox('Blur method', ('Average', 'Gaussian'))
                kernel_size = st.slider('Choose filter size', 1, 19, 3, 2)
                if blur_option == 'Average':
                    image = cv2.blur(image, (kernel_size, kernel_size))
                elif blur_option == 'Gaussian':
                    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            st.write('\n')
            # st.markdown('**Binarize Image**')
            binarize_method = st.selectbox('Binarize method', ('Threshold', 'Adaptive Threshold', 'Edge Detection'))
            if binarize_method == 'Threshold':
                threshold_option = st.selectbox('Threshold method',
                                            ('cv2.THRESH_BINARY', 'cv2.THRESH_BINARY_INV', 
                                                'cv2.THRESH_TRUNC',
                                                'cv2.THRESH_TOZERO', 'cv2.THRESH_TOZERO_INV'))
                threshold_value = st.slider('Threshold value', 0, 255, 100, 1)
                max_value = st.slider('If the pixel exceeds threshold value, change it to', 0, 255, 100, 1)
                
                st.write('\n')
                display_contour = st.checkbox('Draw Contours')
                with col2:
                    (_, image) = cv2.threshold(image, threshold_value, max_value, eval(threshold_option))
                    code = f'''cv2.threshold("{image_file.name}", {threshold_value}, {max_value}, {threshold_option})'''
                    st.image(image, caption=code, use_column_width=True)
                    if display_contour:
                        draw_cnts(img, image)

            elif binarize_method == 'Adaptive Threshold':
                adaptive_method = st.radio('Adaptive kernel method', ('cv2.ADAPTIVE_THRESH_MEAN_C', 'cv2.ADAPTIVE_THRESH_GAUSSIAN_C'))
                threshold_option = st.selectbox('Threshold method',
                                            ('cv2.THRESH_BINARY', 'cv2.THRESH_BINARY_INV', 
                                             'cv2.THRESH_TRUNC',
                                             'cv2.THRESH_TOZERO', 'cv2.THRESH_TOZERO_INV'))
                block_size = st.slider('Kernel size', 1, 19, 3, 2)
                c = st.slider('Constant C value', 0, 10, 1, 1)

                display_contour = st.checkbox('Draw Contours')
                with col2:
                    image = cv2.adaptiveThreshold(image, 255, eval(adaptive_method), eval(threshold_option), block_size, c)
                    code = f'''cv2.adaptiveThreshold("{image_file.name}", 255, {adaptive_method}, {threshold_option}, {block_size}, {c})'''
                    st.image(image, caption=code, use_column_width=True)
                    if display_contour:
                        draw_cnts(img, image)
            
            elif binarize_method == 'Edge Detection':
                edge_option = st.selectbox('Edge detection method',
                                            ('Sobel', 'Laplacian', 'Canny'))
                display_contour = st.checkbox('Draw Contours')
                if edge_option == 'Sobel':
                    with col2: 
                        sobelX = cv2.Sobel(image,cv2.CV_64F, 1, 0)
                        sobelY = cv2.Sobel(image,cv2.CV_64F, 0, 1)

                        sobelX = np.uint8(np.absolute(sobelX))
                        sobelY = np.uint8(np.absolute(sobelY))

                        image = cv2.bitwise_or(sobelX,sobelY)
                        st.image(image, use_column_width=True)

                        if display_contour:
                            draw_cnts(img, image)
                elif edge_option == 'Laplacian':
                    with col2:
                        lap = cv2.Laplacian(image, cv2.CV_64F)
                        image = np.uint8(np.absolute(lap))
                        st.image(image, use_column_width=True)

                        if display_contour:
                            draw_cnts(img, image)
                else:
                    minVal = st.slider('Minimum threshold', 0, 255, 1)
                    maxVal = st.slider('Maximum threshold', 0, 255, 1)
                    with col2: 
                        image = cv2.Canny(image, minVal, maxVal)
                        st.image(image, use_column_width=True)
                        if display_contour:
                            draw_cnts(img, image)


elif choice == 'Blur':
    st.subheader('Affect of Kernel Size: Average and Gaussian')

    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'], key=2)
    if image_file is not None:
        img = load_image(image_file)
        st.image(img, caption='Original', use_column_width=True)
    
        kernel_size = st.slider('Choose a kernel size', 1, 19, 3, 2)
        a_blur = cv2.blur(img, (kernel_size, kernel_size))
        g_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        col1, col2 = st.beta_columns(2)
        with col1:
            code = f"cv2.blur('{image_file.name}', ({kernel_size}, {kernel_size}))"
            st.image(a_blur, caption=code, use_column_width=True)
        with col2: 
            code = f"cv2.GaussianBlur('{image_file.name}', ({kernel_size}, {kernel_size}), 0)"
            st.image(g_blur, caption=code, use_column_width=True)

