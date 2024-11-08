import os
import cv2

# Directory paths
# image_dir = 'train/images'  # Replace with the path to your images
# label_dir = 'train/labels'  # Replace with the path to your labels
# output_dir = 'cropped/train'
# image_dir = 'test/images'  # Replace with the path to your images
# label_dir = 'test/labels'  # Replace with the path to your labels
# output_dir = 'cropped/test'
image_dir = 'valid/images'  # Replace with the path to your images
label_dir = 'valid/labels'  # Replace with the path to your labels
output_dir = 'cropped/valid'

names = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each label file
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        # print(label_file)
        image_name = os.path.splitext(label_file)[0] + '.jpg'  # Assuming images are .jpg; change as needed
        # print(image_name)
        image_path = os.path.join(image_dir, image_name)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_path} not found, skipping...")
            continue
        
        height, width, _ = image.shape
        
        # Read the label file
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as file:
            for line_num, line in enumerate(file):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                
                # Denormalize the coordinates to pixel values
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height
                
                # Calculate top-left and bottom-right corners
                x1 = int(x_center - (box_width / 2))
                y1 = int(y_center - (box_height / 2))
                x2 = int(x_center + (box_width / 2))
                y2 = int(y_center + (box_height / 2))
                
                # Crop the image
                cropped_image = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                
                # Save the cropped image
                piece_type = names[int(class_id)]
                new_output_dir = os.path.join(output_dir, f"{piece_type}")
                # print(new_output_dir)
                os.makedirs(new_output_dir, exist_ok=True)

                # output_path = os.path.join(output_dir, f"/{piece_type}/{os.path.splitext(image_name)[0]}_obj{line_num}")
                output_path = os.path.join(new_output_dir, f"{os.path.splitext(image_name)[0]}_obj{line_num}.jpg")
                cv2.imwrite(output_path, cropped_image)
                print(f"Saved cropped image: {output_path}")
