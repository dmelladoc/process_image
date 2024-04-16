import os
from FindClf import ImageIO
import cv2
from argparse import ArgumentParser

def create_folder(folder:str):
    """
    Creates a folder if it does not exist
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def main(args):
    # Load the image
    image = ImageIO.load_dicom(args.input)
    # Apply CLAHE
    image = ImageIO.clahefusion(image, [1., 2.])
    if args.crop:
        bbox = ImageIO.get_ROIbox(image)
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # Save the image
    create_folder(args.output)
    output_file = os.path.join(args.output, os.path.splitext(os.path.basename(args.input))[0] + '.' + args.ext)

    # As we create a color image, we need to convert it to BGR before saving
    cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



    

if __name__ == '__main__':
    parser = ArgumentParser(prog='Preprocess single image', description="Apply our preprocessing to a DICOM file, returning the processed image")
    parser.add_argument('input', type=str, help='Path to the DICOM file')
    parser.add_argument('--output', '-o', type=str, default='output/',help='Folder to save the processed image, will be created if it does not exist')
    parser.add_argument('--crop', '-c', action='store_true', help='If set, the image will be cropped to the region of interest')
    parser.add_argument('--ext', '-e', type=str, default='png', help='Extension of the processed image. By Default will be a PNG image')
    args = parser.parse_args()

    main(args)