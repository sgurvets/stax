import os
import glob
import rawpy
import cv2
import argparse

def convert_cr2_to_jpeg(input_folder, output_folder):
    # Get a list of all CR2 files in the input folder
    cr2_files = glob.glob(os.path.join(input_folder, '*.CR2'))

    # Process each CR2 file and save as a JPEG
    for cr2_file in cr2_files:
        # Read CR2 file using rawpy
        with rawpy.imread(cr2_file) as raw:
            # Process the raw image
            rgb_image = raw.postprocess()

        # Convert the processed image to a format compatible with OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the BGR image as a JPEG
        output_filename = os.path.splitext(os.path.basename(cr2_file))[0] + '.jpg'
        output_file = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_file, bgr_image)

        print(f'Converted {cr2_file} to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Convert CR2 images to JPEGs.')
    parser.add_argument('input_folder', help='Path to the input folder containing CR2 images')
    parser.add_argument('-o', '--output_folder', default=None, help='Path to the output folder to save JPEG images (default: input folder)')

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = args.input_folder

    convert_cr2_to_jpeg(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main()
