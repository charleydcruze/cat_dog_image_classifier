import os
import random
import shutil

#Set paths
project_dir = os.getcwd()
images_dir = os.path.join(project_dir, "images")

#Fallback: search for "images" directory if nested inside another folder
if not os.path.exists(images_dir):
    for root, dirs, files in os.walk(project_dir):
        if "images" in dirs:
            images_dir = os.path.join(root, "images")
            break

cats_target_dir = os.path.join(project_dir, "dataset", "cats")
dogs_target_dir = os.path.join(project_dir, "dataset", "dogs")

# Target image count per class to eliminate bias 
TARGET_COUNT = 1000 

# Oxford-IIIT Pet cat breed identifiers (lowercase) 
CAT_BREEDS = { "abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau",
               "maine_coon", "persian", "ragdoll", "russian_blue", "siamese", "sphynx", }

print(f"Source images directory: {images_dir}")

if not os.path.exists(images_dir):
    print(
        "Error: Could not find the 'images' folder. Please ensure it is in your project folder."
    )
    exit(1)

#Reset / prepare target dataset directories
if os.path.exists(cats_target_dir):
    shutil.rmtree(cats_target_dir) 

if os.path.exists(dogs_target_dir): 
    shutil.rmtree(dogs_target_dir)

os.makedirs(cats_target_dir, exist_ok=True) 
os.makedirs(dogs_target_dir, exist_ok=True)

#Categorise all available images into lists
all_cat_files = [] 
all_dog_files = []

for filename in os.listdir(images_dir): 
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")): 
        continue
    # Extract breed string (e.g. 'Abyssinian\_100.jpg' -&gt; 'abyssinian'
    breed_part = filename.rsplit("_", 1)[0].lower()

    # Oxford dataset convention: Cat filenames start with capital letters or match breed list
    if breed_part in CAT_BREEDS or filename[0].isupper():
        all_cat_files.append(filename)
    else:
        all_dog_files.append(filename)

print(
    f"🔍 Found {len(all_cat_files)} total cat images and {len(all_dog_files)}"
     " total dog images." 
)

#Set seed for reproducible random sampling 
random.seed(42)

#Ensure we have enough images for 1000 each
selected_cats = ( 
    random.sample(all_cat_files, TARGET_COUNT) 
    if len(all_cat_files) >= TARGET_COUNT 
    else all_cat_files 
) 

selected_dogs = ( 
    random.sample(all_dog_files, TARGET_COUNT) 
    if len(all_dog_files) >= TARGET_COUNT 
    else all_dog_files 
)

#Copy sampled images to destination folders
print( 
    f"Copying {len(selected_cats)} cat images and {len(selected_dogs)} dog" 
    " images..." 
)

for filename in selected_cats: 
    shutil.copy( 
        os.path.join(images_dir, filename), 
        os.path.join(cats_target_dir, filename), 
    ) 

for filename in selected_dogs: 
    shutil.copy( 
        os.path.join(images_dir, filename), 
        os.path.join(dogs_target_dir, filename), 
    )

print("\nDataset balancing complete!")