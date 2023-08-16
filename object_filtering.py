import os

# Path to the ImageSets/Main directory
main_dir = "D:/dataset/PASCAL_VOC_2007/VOC2007/ImageSets/Layout"

# Path to the Annotations directory
annotations_dir = "D:/dataset/PASCAL_VOC_2007/VOC2007/Annotations"

# List of image filenames with one object
filtered_images = []

# Read the train.txt file that contains the image filenames for training
with open(main_dir+"/train.txt", "r") as train_file:
    train_images = train_file.read().splitlines()

# Iterate over the train images
for image in train_images:
    # Read the corresponding XML annotation file
    annotation_file = os.path.join(annotations_dir, image + ".xml")
    with open(annotation_file, "r") as xml_file:
        annotation_data = xml_file.read()
        
    # Count the number of object tags in the XML file
    num_objects = annotation_data.count("<object>")
    
    # If there is exactly one object, add the image to the filtered list
    if num_objects == 1:
        filtered_images.append(image)

# Print the filtered image filenames
for image in filtered_images:
    print(image)
