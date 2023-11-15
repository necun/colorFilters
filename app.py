from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi import FastAPI
import random
import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import io

app = FastAPI()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def apply_sharp_black_filter(image_path):
    img = cv2.imread(image_path)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def apply_vibrant_filter(image_path, saturation_factor=2.0, brightness_factor=1.2):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    vibrant_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return vibrant_image

def apply_soft_tone_filter(image_path, gamma=1.5):
    img = cv2.imread(image_path)
    soft_tone_image = np.power(img / 255.0, gamma) * 255.0
    soft_tone_image = soft_tone_image.astype(np.uint8)
    return soft_tone_image

def apply_black_pop_filter(image_path, brightness_factor=1.2, contrast_factor=1.2):
    img = cv2.imread(image_path)
    img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_factor)
    return img

def apply_sepia_filter(image_path):
    img = cv2.imread(image_path)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(img, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

def grayscale(image_path):
    color_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

@app.get("/")
def root():
    return {"message": "Welcome to the image processing API!"}

@app.post("/upload/")
# async def upload_image(file: UploadFile):
#     # Save the uploaded file
#     # with open("C:/Users/DELL/OneDrive/Desktop/", "wb") as image_file:
#     #     image_file.write(file.file.read())
#     img = file.file.read()
#     img = Image.open(img)
#     # return {"filename": file.filename}
async def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        # Save the uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        image.save(upload_path)
        return {"message": "Upload successful", "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

# @app.get("/sharp_black_filter")
# def sharp_black_filter(image_path: str):
#     processed_image = apply_sharp_black_filter("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
#
# @app.get("/vibrant_filter")
# def vibrant_filter(image_path: str):
#     processed_image = apply_vibrant_filter("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
#
# @app.get("/soft_tone_filter")
# def soft_tone_filter(image_path: str):
#     processed_image = apply_soft_tone_filter("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
#
# @app.get("/black_pop_filter")
# def black_pop_filter(image_path: str):
#     processed_image = apply_black_pop_filter("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
#
# @app.get("/sepia_filter")
# def sepia_filter(image_path: str):
#     processed_image = apply_sepia_filter("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
#
# @app.get("/grayscale")
# def grayscale(image_path: str):
#     processed_image = grayscale("uploaded_image.jpg")
#     cv2.imwrite("C:/Users/DELL/Downloads/processed_image.jpg", processed_image)
#     return FileResponse("processed_image.jpg")
@app.get("/process/{filter_name}")
async def process_image(filter_name: str, filename: str):
    try:
        # Load the uploaded image
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        image = Image.open(upload_path)

        # Apply the selected filter
        if filter_name == "sharp_black":
            processed_image = apply_sharp_black_filter(image)
        elif filter_name == "vibrant":
            processed_image = apply_vibrant_filter(image)
        elif filter_name == "soft_tone":
            processed_image = apply_soft_tone_filter(image)
        elif filter_name == "black_pop":
            processed_image = apply_black_pop_filter(image)
        elif filter_name == "sepia":
            processed_image = apply_sepia_filter(image)
        elif filter_name == "grayscale":
            processed_image = grayscale(image)
        else:
            return {"error": "Invalid filter name"}

        # Save the processed image
        processed_filename = f"{filter_name}_{filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        processed_image.save(processed_path)

        return FileResponse(processed_path, media_type="image/jpeg", headers={"Content-Disposition": f"attachment; filename={processed_filename}"})
    except Exception as e:
        return {"error": str(e)}
