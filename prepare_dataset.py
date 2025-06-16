import os
import shutil
import splitfolders

# Step 0: Install split-folders if not already installed
# Run this in your terminal or notebook before running the script:
# pip install split-folders
print("[INFO] Script started...")
# Step 1: Paths
raw_dataset_path = "PlantVillage"            # Folder from Kaggle
cleaned_dataset_path = "CleanedDataset"      # Temp cleaned folder
output_dataset_path = "dataset"              # Final train/val structure


# Step 2: Clean and Copy Folders
if not os.path.exists(cleaned_dataset_path):
    os.makedirs(cleaned_dataset_path)

print("[INFO] Cleaning and copying class folders...")
for folder in os.listdir(raw_dataset_path):
    folder_path = os.path.join(raw_dataset_path, folder)
    if os.path.isdir(folder_path):
        # Clean label from folder name (e.g., "Tomato___Early_blight" → "Early_blight")
        cleaned_label = folder.split("___")[-1]
        target_path = os.path.join(cleaned_dataset_path, cleaned_label)

        if not os.path.exists(target_path):
            shutil.copytree(folder_path, target_path)
        else:
            # If the folder already exists (e.g., Healthy from Tomato and Potato), merge images
            for file in os.listdir(folder_path):
                src_file = os.path.join(folder_path, file)
                dst_file = os.path.join(target_path, file)
                shutil.copy2(src_file, dst_file)

print("[INFO] Splitting into train and val folders...")
# Step 3: Split into train/val folders
splitfolders.ratio(
    input=cleaned_dataset_path,
    output=output_dataset_path,
    seed=42,
    ratio=(0.8, 0.2),
    group_prefix=None
)

print("[✅ DONE] Dataset ready in 'dataset/train' and 'dataset/val'.")
