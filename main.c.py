import matplotlib.pyplot as plot
from PIL import Image

from preprocessing.data_load import Dalaloader





def main():
    """ Load the data """
    print("--- Step 1: Data loading ---")
    Loader = Dalaloader(data_path="./dataset")
    collected_data = Loader.collect_from_folder(num_folders=9)
    print(f"Loaded {len(collected_data)} images")

    image_arr = collected_data[0]["images"][0]
    label = collected_data[0]["label"][0]

    image = Image.fromarray(image_arr)

    print(collected_data[0]["images"][0].shape)

    plot.imshow(image)
    plot.title(label)
    plot.axis('off')
    plot.show()


if __name__ == "__main__":
    main()
