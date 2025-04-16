import os
import shutil
from glob import glob

root = "/DATA/Rajes/BCD/VOC"
dataset_name = "INBreast"
dataset_path = os.path.join(root, dataset_name)                     # /DATA/Rajes/BCD/VOC/DDSM
SF_dataset_path = f"{dataset_path}_sf"                              # /DATA/Rajes/BCD/VOC/DDSM_sf
source_annotations = "../Source/output/vgg16/PseudoL_ddsm2rsna/Annotations"

# Check if required paths exist
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
if not os.path.exists(source_annotations):
    raise FileNotFoundError(f"Source annotations not found: {source_annotations}")

# Create SF_dataset_path if it doesn't exist
os.makedirs(SF_dataset_path, exist_ok=True)

# Copy JPEGImages and ImageSets from original dataset
for folder in ["JPEGImages", "ImageSets"]:
    src = os.path.join(dataset_path, "VOC2007", folder)
    dst = os.path.join(SF_dataset_path, "VOC2007", folder)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Required folder not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

# Copy Source Annotations to SF_dataset_path/VOC2007/Annotations
dst_annotations = os.path.join(SF_dataset_path, "VOC2007/Annotations")
os.makedirs(dst_annotations, exist_ok=True)

for file in glob(os.path.join(source_annotations, "*.xml")):
    shutil.copy2(file, dst_annotations)

# Count of source annotations
source_count = len(glob(os.path.join(source_annotations, "*.xml")))

# Copy test annotations from original dataset (assumed to be those not in source)

# original_annotations = os.path.join(dataset_path, "VOC2007", "Annotations")
# if not os.path.exists(original_annotations):
#     raise FileNotFoundError(f"Original Annotations folder not found: {original_annotations}")

# for file in glob(os.path.join(original_annotations, "*.xml")):
#     filename = os.path.basename(file)
#     if not os.path.exists(os.path.join(dst_annotations, filename)):
#         shutil.copy2(file, dst_annotations)

# # Count of test annotations (files from original that were not in source)
# test_annotation_files = [
#     f for f in glob(os.path.join(original_annotations, "*.xml"))
#     if not os.path.exists(os.path.join(source_annotations, os.path.basename(f)))
# ]
# test_count = len(test_annotation_files)

# Output counts
print(f"{SF_dataset_path} created")
print(f"Count of Source Annotations: {source_count}")
# print(f"Count of Test Annotations: {test_count}")
