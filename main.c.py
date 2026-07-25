import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

from preprocessing.data_load import Dalaloader
from preprocessing.data_processing import DataProcessing
from model.model_cnn import ModelCNN



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
            normalize_image = preprocessor.img_normalize(gray_img)
            images_list[image_idx] = normalize_image


    """
        Data Augmentation
    """

    dataset_train_x   = []
    dataset_train_y   = []

    dataset_test_x = []
    dataset_test_y  = []

    image_set_array = loader.image_set_creation(collected_data)
    label_set_array = loader.label_set_creation(collected_data)

    dataset_train_x, dataset_test_x, dataset_train_y, dataset_test_y = train_test_split(
        image_set_array,
        label_set_array,
        test_size=0.2,
        random_state=42,
        stratify=label_set_array
    )

    print(f"Shape: {dataset_train_x.shape}")

    model = ModelCNN()

    model.build_model()
    model.train_model(dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y)


    # for images in collected_data:
    #
    #     lable = images["label"]
    #     images = images["images"]
    #
    #     for idx, image in enumerate(images):
    #         if (idx == 0):
    #             print(f"Shape: {image[0,0]}")
    #             plot.imshow(image,cmap="gray")
    #             plot.show()


if __name__ == "__main__":
    main()
