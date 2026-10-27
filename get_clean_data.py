import os
import shutil
import torchvision.datasets as datasets

print("Downloading clean Oxford-IIIT Pet dataset...")

# Download official dataset via PyTorch
pet_dataset = datasets.OxfordIIITPet( root="oxford\_pets", split="trainval", target_types="category", download=True ) 
# Oxford-IIIT Pet cat breeds list 
cat_breeds = { "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau",
               "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx", }

# Create target directories 
os.makedirs("dataset/cats", exist_ok=True) 
os.makedirs("dataset/dogs", exist_ok=True) 

# Sort images into cats and dogs 
cat_count = 0 
dog_count = 0 

for img_path, _ in zip(pet_dataset._images, pet_dataset._targets): 
    filename = os.path.basename(img_path) 
    # Extract breed name from filename (e.g. 'cat_101.jpg' or 'english_setter_12.jpg') 
    breed_name = filename.rsplit("_", 1)[0] 
    
    if breed_name in cat_breeds:
        shutil.copy(img_path, os.path.join("dataset/cats", filename)) 
        cat_count += 1 
    else: 
        shutil.copy(img_path, os.path.join("dataset/dogs", filename)) 
        dog_count += 1 
        
print( 
    f"Done. Populated dataset/cats ({cat_count} images) and dataset/dogs" 
    f" ({dog_count} images)." 
)