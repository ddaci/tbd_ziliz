import os
import json
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to Tiny ImageNet images folder
image_folder = r"D:\\daci\\tbdproject\\archive\\tiny-imagenet-200\\test\\images"

# Load pre-trained MobileNetV2 model (exclude top layer for feature extraction)
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(64, 64, 3))

# Function to process and extract vectors
def extract_feature_vectors(image_folder, target_size=(64, 64)):
    feature_vectors = []
    file_names = []
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if os.path.isfile(file_path):
            # Load and preprocess the image
            img = load_img(file_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)

            # Extract feature vector
            feature_vector = model.predict(img_array)
            feature_vectors.append(feature_vector.flatten())
            file_names.append(file_name)
    return np.array(feature_vectors), file_names

# Extract feature vectors
vectors, file_names = extract_feature_vectors(image_folder)

# Save feature vectors
output_path = os.path.join(os.path.dirname(image_folder), "feature_vectors_mobilenet.npy")
np.save(output_path, vectors)
print(f"Feature vectors saved to: {output_path}")

# Generate and save label-to-image map
label_to_image_map = {f"label_{i}": file_name for i, file_name in enumerate(file_names)}
map_path = os.path.join(os.path.dirname(image_folder), "label_to_image_map.json")
with open(map_path, "w") as f:
    json.dump(label_to_image_map, f)
print(f"Label-to-image map saved to: {map_path}")
