import os
import shutil
from PIL import Image
from torchvision.datasets import CIFAR10

#Setup the target directory.
target_dir = os.path.join("dataset", "other")
os.makedirs(target_dir, exist_ok=True)

print("Downloading general dataset (CIFAR10)")
dataset = CIFAR10(root="./temp_data", train="true", download=True)

#General class indices: airplane (0), automobile (1), ship (8), truck(9)
non_pet_classes = {0, 1, 8, 9}

count = 0
target_count = 1000

print("Extracting 1000 non-cat/dog images into other dataset.")

for i, (img, label) in enumerate(dataset):
    if label in non_pet_classes:
        #Save as jpgs
        img_path = os.path.join(target_dir, f"other_{count + 1:04d}.jpg")
        img.convert("RGB").save(img_path)
        count +=1

        if count >= target_count:
            break

#Clean up temporary download folder.
if os.path.exists("./temp_data"):
    shutil.rmtree("./temp_data")

print(f"Successfully saved {count} images into dataset/other.")