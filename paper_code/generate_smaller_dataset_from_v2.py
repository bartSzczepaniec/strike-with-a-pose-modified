# This script generates a smaller dataset (custom_dataset_v4) by choosing 100 images of each object
# from custom_dataset_v3 containing 4k validation images per class and 1k training object of each class
import os
import shutil


def get_files(dir):
    return [os.path.join(dir, x) for x in os.listdir(dir)]

def generate_smaller_dataset(src_dataset_dir, new_dataset_dir, images_per_object):
    for file in os.listdir(src_dataset_dir):
        src_file = os.path.join(src_dataset_dir, file)
        new_file = os.path.join(new_dataset_dir, file)
        img_number = int(file.split("_")[2].removeprefix("i"))
        if img_number < images_per_object and os.path.isfile(src_file):
            shutil.copy2(src_file, new_file)
            print(src_file)


dataset_v3_dir = "./custom_dataset_v2"
new_dataset_dir = "./custom_dataset_v3"

os.makedirs(new_dataset_dir, exist_ok=True)
generate_smaller_dataset(dataset_v3_dir, new_dataset_dir, 100)





