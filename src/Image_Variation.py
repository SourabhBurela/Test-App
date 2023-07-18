import streamlit as st
import openai
import requests
from openai import Image
from PIL import Image
import base64
from io import BytesIO
import multiprocessing
from multiprocessing import Pool

def generate_download_button(image, filename):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = buffered.getvalue()
    base64_encoded = base64.b64encode(image_data).decode()
    download_button = f'<a href="data:file/png;base64,{base64_encoded}" download="{filename}">Download</a>'
    return download_button

def generate_variation(image_data):
    response = openai.Image.create_variation(
        image=image_data,
        n=1,
        size="1024x1024"
    )
    return response["data"][0]["url"]

def generate_images_parallel(image_data, num_images):
    with Pool() as pool:
        generated_urls = pool.map(generate_variation, [image_data] * num_images)
    return generated_urls

def page2():
    st.title("OpenAI DALL·E Image Variation")

    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg'])
    generate_button = st.button("Generate")

    if generate_button:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_data = BytesIO()
            image.save(image_data, format="PNG")
            image_data = image_data.getvalue()
            num_images = 10
            st.image(image, caption="Uploaded image", use_column_width=True)

            generated_urls = generate_images_parallel(image_data, num_images)

            cols = st.columns(5)
            for idx, url in enumerate(generated_urls):
                image = Image.open(requests.get(url, stream=True).raw)
                cols[idx % 5].image(image, caption=f"Generated image: {idx+1}", use_column_width=True)
                download_button = generate_download_button(image, f"Generated_Image_{idx+1}.png")
                cols[idx % 5].write(download_button, unsafe_allow_html=True)