import os
import shutil
import re

# Define the root directory where all the class folders are located
root_dir = "./texture/cue-conflict/"

# Get all the class directories in the root directory
class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Go through each directory
for current_dir in class_dirs:
    current_dir_path = os.path.join(root_dir, current_dir)
    
    # Get all image files in the current directory
    for filename in os.listdir(current_dir_path):
        if filename.endswith(".png"):
            # Split the filename to get the second class name (after the dash)
            second_class = filename.split("-")[1].split(".")[0]
            
            # Remove any trailing numbers from the second class
            second_class = re.sub(r'\d+$', '', second_class)
            
            # Check if the second class directory exists
            target_dir = os.path.join(root_dir, second_class)
            if second_class in class_dirs:
                # Move the file to the corresponding second class directory
                source_path = os.path.join(current_dir_path, filename)
                target_path = os.path.join(target_dir, filename)
                
                # Move the file
                shutil.move(source_path, target_path)
                print(f"Moved {filename} from {current_dir} to {second_class}")
            else:
                print(f"Directory {second_class} does not exist. Skipping {filename}.")
