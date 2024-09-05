import cv2
import os
import numpy as np

def pencil_sketch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        img_rgb = cv2.merge((b, g, r))
    else:
        img_rgb = img
        a = None
    
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpening)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    gaussgray = cv2.GaussianBlur(inv, ksize=(15, 15), sigmaX=0, sigmaY=0)
    
    def dodgeV2(image, mask):
        return cv2.divide(image, 255 - mask, scale=256)
    
    pencil_img = dodgeV2(gray, gaussgray)
    pencil_img = cv2.cvtColor(pencil_img, cv2.COLOR_GRAY2BGR)
    
    white_bg = np.ones_like(pencil_img) * 200
    
    opacity = 150 / 255.0
    output_img = cv2.addWeighted(pencil_img, opacity, white_bg, 1 - opacity, 0)
    
    if a is not None:
        output_img = cv2.merge((output_img, a))
    
    return output_img

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            sketch = pencil_sketch(image_path)
            if sketch is not None:
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, sketch)
                print(f"Processed {filename}")

input_directory = './dumps'
output_directory = './replacements'

process_directory(input_directory, output_directory)