import matplotlib.pyplot as plot

from preprocessing.data_load import Dalaloader
from preprocessing.data_processing import DataProcessing




def main():
    """
        Load the data from the dataset
    """
    print("Step 1: Data loading ---")
    loader = Dalaloader(data_path="./dataset")
    collected_data = loader.collect_from_folder(num_folders=9)


    """ 
        Preprocessing all the data/images
    """
    print("Step 2: Data Preprocessing ---")
    preprocessor = DataProcessing()

    for folders in collected_data:
        label       = folders["label"]
        images_list = folders["images"]

        print(f"Dataset Label: {label}")

        for image_idx, image in enumerate(images_list):
            gray_img = preprocessor.img_convert_to_greyscale(image)
            images_list[image_idx] = gray_img





    for images in collected_data:

        lable = images["label"]
        images = images["images"]

        for idx, image in enumerate(images):
            if (idx == 0):
                print(f"Shape: {image[0,0]}")
                plot.imshow(image,cmap="gray")
                plot.show()


if __name__ == "__main__":
    main()
