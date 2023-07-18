import streamlit as st
import openai
import requests
from openai import Image
from PIL import Image
import base64
from io import BytesIO
import multiprocessing
from multiprocessing import Pool
from PIL import Image as PILImage, ImageFilter

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        #model="image-alpha-001",
        size="1024x1024",
        response_format="url"
    )
    if "data" not in response or not response["data"]:
        st.error(f"Image generation failed for prompt: {prompt}. Please try again.")
        return None

    image_url = response["data"][0]["url"]
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

def apply_filter(image, filter_name):
    if filter_name == "Blur":
        return image.filter(ImageFilter.BLUR)
    elif filter_name == "Contour":
        return image.filter(ImageFilter.CONTOUR)
    elif filter_name == "Detail":
        return image.filter(ImageFilter.DETAIL)
    elif filter_name == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_name == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_name == "Find Edges":
        return image.filter(ImageFilter.FIND_EDGES)
    elif filter_name == "Smooth":
        return image.filter(ImageFilter.SMOOTH)
    elif filter_name == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_name == "Gaussian Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_name == "Box Blur":
        return image.filter(ImageFilter.BoxBlur(radius=2))
    else:
        return image
    
    
def generate_images_parallel(prompts, num_images,filter_name):
    num_gpus = multiprocessing.cpu_count()  # Number of available GPUs
    num_processes = min(num_gpus, num_images)  # Number of processes to run in parallel

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(generate_image, prompts[:num_processes])
    pool.close()
    pool.join()

    return results

def generate_images(captions, num_images,filter_name):
    images = []
    num_prompts = (num_images + multiprocessing.cpu_count() - 1) // multiprocessing.cpu_count()
    for i in range(num_prompts):
        start_idx = i * multiprocessing.cpu_count()
        end_idx = min((i + 1) * multiprocessing.cpu_count(), num_images)

        prompts = captions[start_idx:end_idx]
        results = generate_images_parallel(prompts, end_idx - start_idx,filter_name)

        for result in results:
            if result is not None:
                images.append(result)

    if not images:
        return

    # Displaying images in grid format
    num_cols = 5
    num_rows = (num_images - 1) // num_cols + 1

    cols = st.columns(num_cols)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if idx < num_images:
                cols[j].image(images[idx], caption=f"{captions[idx]} - Image {idx+1}", use_column_width=True)

                # download button
                download_button = generate_download_button(images[idx], f"{captions[idx]}_Image_{idx+1}.png")
                cols[j].markdown(download_button, unsafe_allow_html=True)

                idx += 1
            else:
                break

def generate_download_button(image, filename):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = buffered.getvalue()
    base64_encoded = base64.b64encode(image_data).decode()
    download_button = f'<a href="data:file/png;base64,{base64_encoded}" download="{filename}">Download</a>'
    return download_button

def page1():
    st.title("Image Generation using DALL-E API")
    caption = st.text_area("Enter your text prompt:", value="", height=100, max_chars=500)
    num_images = 20
    
    filter_name = st.selectbox("Apply Filter:", ["None", "Blur", "Contour", "Detail", "Edge Enhance", "Emboss",
                                                 "Find Edges", "Smooth", "Sharpen", "Gaussian Blur", "Box Blur"])
    if st.button(f"Generate"):
        captions = [caption] * num_images
        generate_images(captions, num_images,filter_name)