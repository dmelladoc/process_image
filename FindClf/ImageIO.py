import os
import cv2
import numpy as np
from typing import Sequence, Tuple, Any

import pydicom as dcm
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_image(path:str):
    """
    Loads an image from provided path
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _normalization(image):
    return (image - image.min()) / (image.max() - image.min())


def _apply_clahe(image:np.ndarray, thresh=1.):
    """
    Applies CLAHE to an image
    Args:
        image: image to apply CLAHE
        thresh: threshold for CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=thresh)
    return clahe.apply(image)


def load_dicom(path:str) -> np.ndarray:

    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    
    with dcm.dcmread(path) as ds:
        # apply VOI LUT
        image = apply_voi_lut(ds.pixel_array, ds)
        # check if image is inverted
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            if image.dtype in [np.uint8, np.uint16, np.int8, np.int16]:
                image = np.iinfo(image.dtype).max - image
            elif image.dtype in [np.float32, np.float64]:
                image = image.max() - image
        
        image = _normalization(image)
        # Should we assume we always want an uint8 image?
        return  (255 * image).astype(np.uint8)


def get_ROIbox(image:np.ndarray):
    """
    Obtains the bounding box of the Region of Interest in the image
    Args:
        image: image to process
    Returns:
        bbox: bounding box of the Region of Interest
    """

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(breast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([x, y, x + w, y + h], dtype=np.int64)


def load_and_crop(image_path:str, apply_fusion=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and crops an image
    Args:
        image_path: path to the image
    Returns:
        cropped: cropped image
        bbox: bounding box of the Region of Interest
    """
    # If it is a dicom image, load it with the dicom library
    if os.path.splitext(image_path)[1] in ['.dicom', '.dcm']:
        image = load_dicom(image_path)
    else:
        #otherwise we assume its a generic image and open it with openCV
        image = load_image(image_path)

    if apply_fusion:
        image = clahefusion(image, [1.0, 2.0])
    bbox = get_ROIbox(image)
    cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return cropped, bbox


def clahefusion(image:np.ndarray, thresholds:Sequence[float]):
    """
    Apply CLAHE to an image with different thresholds, creating a fused image
    Args:
        image: image to apply CLAHE
        thresholds: list of thresholds for CLAHE
    Returns:
        fused_image: image with CLAHE applied
    """
    if not len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    height, width = image.shape
    bg = image == 0 #Mask for the background

    list_of_images = [image]
    for thresh in thresholds:
        clahed = _apply_clahe(image, thresh)
        clahed[bg] = 0
        list_of_images.append(clahed)
    
    fused_image = cv2.merge(list_of_images)
    return fused_image