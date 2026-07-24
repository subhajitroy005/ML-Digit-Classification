from pathlib import Path as dir
from PIL import Image
import numpy as np


num_of_folders = 10

# Structure array
dataset_structure = []

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]



for i in range(1, num_of_folders + 1):
    # get the folder names
    folder_path = dir(f"./dataset/{i}")

    # Directory wise images put inside a structure in lables
    if not (folder_path.is_dir()):
        continue

    #images array container
    images_list = []

    # Loop through files and put inside the structure
    for files in folder_path.iterdir():
        if files.suffix.lower() in IMAGE_EXTENSIONS:
            img = Image.open(files)
            img_arr = np.array(img)
            images_list.append(img_arr)

    #Now append to the strucre
    dataset_structure.append( {"label": "{i}", "images": images_list } )

