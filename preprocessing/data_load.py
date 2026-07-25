from pathlib import Path
import numpy as np
from PIL import Image


class Dalaloader:
    """ Class responsible for data load from dataset """

    def __init__(self, data_path="./dataset", extensions=".png"):
        self.base_path = Path(data_path)
        if extensions is None:
            self.extensions = {".png"}
        else:
            self.extensions = extensions

    def collect_from_folder(self, num_folders):
        """ Loop through dataset folders and collect images  """

        dataset_struct = []

        for i in range(0, num_folders + 1):
            folder_path = self.base_path/str(i)

            if not(folder_path.exists()):
                print(f"Folder not found {folder_path}")
                continue

            images_list = []

            for file_path in folder_path.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    try:
                        img = Image.open(file_path)
                        img_arr = np.array(img)
                        images_list.append(img_arr)
                    except Exception as e:
                        print(f"Error loading {file_path}:{e}")

            dataset_struct.append({"label":str(i) , "images":images_list})
            # Lop ends


        return dataset_struct