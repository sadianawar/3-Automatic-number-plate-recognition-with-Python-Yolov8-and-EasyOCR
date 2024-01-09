# output_dir='./output_images'

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import pandas as pd

def visualize_results(image_path, results_path, output_dir='./output_images'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the original image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Specify the output image path with the original image name
    output_image_path = os.path.join(output_dir, f'{image_name}_output_visualized_image.jpg')

    # Load the image
    image = cv2.imread(image_path)

    # Read results from CSV file
    results_df = pd.read_csv(results_path)

    # Iterate over the rows in the CSV file
    for index, row in results_df.iterrows():
        # Extract bounding box coordinates
        bbox_str = row['license_plate_bbox']
        bbox = [float(coord) for coord in bbox_str[1:-1].split()]  # Convert to float

        # Draw bounding box on the image
        bbox = [int(coord) for coord in bbox]  # Convert back to int
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Extract license plate information
        license_plate_text = row['license_number']

        # Draw a larger white box with increased font size and thickness for the license plate text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0  # Increase font size
        font_thickness = 4  # Increase font thickness
        text_spacing = 10  # Increase spacing between letters

        # Get the size of the text
        text_size = cv2.getTextSize(license_plate_text, font, font_scale, font_thickness)[0]

        # Calculate the position of the text and the bounding box
        text_x = bbox[0] + (bbox[2] - bbox[0]) // 2 - text_size[0] // 2
        text_y = bbox[3] + text_size[1] + text_spacing  # Adjust the y-coordinate

        # Calculate the position and size of the white box
        box_top_left = (bbox[0] - 10, bbox[3] + 3)
        box_bottom_right = (bbox[2] + 10, text_y + text_size[1] - 5)

        # Draw the larger white box
        cv2.rectangle(image, box_top_left, box_bottom_right, (255, 255, 255), -1)

        # Draw the license plate text with increased spacing, font size, and thickness
        cv2.putText(image, license_plate_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness,
                    cv2.LINE_AA)

    # Save the image with bounding boxes and formatted text
    cv2.imwrite(output_image_path, image)

if __name__ == "__main__":
    # Specify the path to the original image, the CSV file with results, and the output image path
    image_path = 'G:/Car Number Plate Detection/DATASETS/test dataset/UK License Plate/car 14.jpg'
    results_csv_path = './image_output_test.csv'

    # Visualize the results and save the image
    visualize_results(image_path, results_csv_path)
