from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi import FastAPI
import random
import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import io
from fastapi.responses import JSONResponse

app = FastAPI()

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def apply_sharp_black_filter(image_path):
    print(image_path)
    img = cv2.imread(image_path)
    print('9090')
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])


    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

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

def apply_brightness_adjustment(image_path):
    img = cv2.imread(image_path)
    adjust=50
    adjust = max(adjust, -100)
    img = cv2.convertScaleAbs(img, alpha=1, beta=adjust)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

    
@app.get("/")
def root():
    return {"message": "Welcome to the image processing API!"}
@app.post("/uploadAndProcess/{filter_name}")
# async def upload_image(file: UploadFile):
#     # Save the uploaded file
#     # with open("C:/Users/DELL/OneDrive/Desktop/", "wb") as image_file:
#     #     image_file.write(file.file.read())
#     img = file.file.read()
#     img = Image.open(img)
#     # return {"filename": file.filename}
async def upload_image(filter_name: str, file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        # Save the uploaded file
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(file.filename)
        image.save(upload_path)
        image = Image.open(upload_path)
        print(image)
        # processImage(filter_name,upload_path)
        # Apply the selected filter
        if filter_name == "sharp_black":
            processed_image = (apply_sharp_black_filter(upload_path))
            
        elif filter_name == "soft_tone":
            processed_image = (apply_soft_tone_filter(upload_path))
        elif filter_name == "black_pop":
            processed_image = (apply_black_pop_filter(upload_path))
        elif filter_name == "sepia":
            processed_image = (apply_sepia_filter(upload_path))
        elif filter_name == "grayscale":
            processed_image = (grayscale(upload_path))  
        elif filter_name == "bright":
            processed_image = (apply_brightness_adjustment(upload_path))  
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filter name",
            )
        # Save the processed image
        processed_filename = f"{filter_name}_{file.filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        print('hi')
        print(processed_image)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        print(processed_path)
        print('bye')
        # processed_image.save(processed_path)
        cv2.imwrite(processed_path,processed_image)
        return {"message": "Upload successful", "filename": file.filename}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    # except Exception as e:
    #     return {"error": str(e)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


