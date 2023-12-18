import cv2
import numpy as np


def apply_sharp_black_filter(image_path):
    img = cv2.imread(image_path)
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

def apply_contrast_adjustment(image_path):
    img = cv2.imread(image_path)
    adjust = 50
    adjust = max(adjust, -100)
    contrast_factor = np.power((adjust + 100) / 100, 2)
    img = ((img / 255 - 0.5) * contrast_factor + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def apply_brightness_adjustment(image_path):
    img = cv2.imread(image_path)
    adjust=50
    adjust = max(adjust, -100)
    img = cv2.convertScaleAbs(img, alpha=1, beta=adjust)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def adjust_hue(image_path):
    original_image = cv2.imread(image_path)
    adjust = 30
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + adjust) % 180
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def adjust_saturation(image_path):
    original_image = cv2.imread(image_path)
    adjust = 20
    adjust_factor = adjust * -0.01
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    max_array = np.max(rgb_image, axis=-1, keepdims=True)
    sat_array = np.where(
        rgb_image != max_array,
        rgb_image + (max_array - rgb_image) * adjust_factor,
        rgb_image,
    )
    adjusted_image = cv2.cvtColor(sat_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return adjusted_image
    
if __name__ == "__main__":
    image_path = "Original Image Before applying filters.jpg"
    apply_sharp_black_filter(image_path)
    apply_soft_tone_filter(image_path)
    apply_black_pop_filter(image_path, brightness_factor=1.2, contrast_factor=1.2)
    apply_sepia_filter(image_path)
    grayscale(image_path)
    apply_contrast_adjustment(image_path)
    apply_brightness_adjustment(image_path)
    adjust_hue(image_path)
    adjust_saturation(image_path)
