import os
import re
import shutil
from PIL import Image

def rename_and_copy_images(source_directory, destination_directory):
    source_directory.replace("\\", "/")
    print(source_directory)


    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Get the directory name we are working with
    directory_name = os.path.basename(source_directory)

    # Extract the date and SAMSON number from the directory name with this regex (\d{4}-\d{2}-\d{2}.*?)_.*(SAMSON\d{1})_.*
    match = re.match(r"(\d{4}-\d{2}-\d{2}.*?)_.*(SAMSON\d{1})_.*", directory_name)
    if match:
        date = match.group(1)
        samson = match.group(2)
    else:
        print("Directory name does not match the expected format")
        return

    # Thats the basename now. Now we go into the directory, get every file and put that date and samson before the img number
    for root, dirs, files in os.walk(source_directory):
        for file_name in files:
            # Get the file extension
            file_extension = file_name.split('.')[-1]

            # Create the new file name
            new_file_name = f"{date}_{samson}_{file_name}"
            new_path = os.path.join(destination_directory, new_file_name)

            # Open the image using PIL
            image_path = os.path.join(root, file_name)
            image = Image.open(image_path)

            # Rotate the image by 90 degrees clockwise
            rotated_image = image.rotate(-90, expand=True)


            # Save the rotated image to the destination directory with the new file name
            rotated_image.save(new_path)

            print(f"Copied, rotated, and renamed: {file_name} -> {new_file_name}")

# Specify the source directory where the folders and images are located
source_directory1 = 'D:\Bachelor\Daten/2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581'
source_directory2 = 'D:\Bachelor\Daten/2024-04-15_11-15-56_D12_rot_Bluete_SAMSON3_1713172557'
source_directory3 = "D:\Bachelor\Daten/2024-04-15_16-32-13_Thore_richtig_SAMSON3_1713191533"


# Specify the destination directory where the renamed images will be saved
destination_directory = "D:/Bachelor/Daten/ALL_DATA"

#print the size of the source directory
print("Size before: " + str(len(os.listdir(destination_directory))))

# Call the function to rename and copy the images
rename_and_copy_images(source_directory1, destination_directory)
rename_and_copy_images(source_directory2, destination_directory)
rename_and_copy_images(source_directory3, destination_directory)

#print the size of the source directory
print("Size after: " + str(len(os.listdir(destination_directory))))
