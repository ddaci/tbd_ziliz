from pymilvus import MilvusClient
import json
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import shutil
import time
from dotenv import load_dotenv



load_dotenv()  # Încarcă variabilele din fișierul .env

CLUSTER_ENDPOINT = os.getenv("CLUSTER_ENDPOINT")
TOKEN = os.getenv("TOKEN")

if not CLUSTER_ENDPOINT or not TOKEN:
    raise ValueError("Variabilele din fișierul .env nu sunt setate corect!")

# Configurații
COLLECTION_NAME = "tiny_imagenet_vectors"
image_folder = r"D:\\daci\\tbdproject\\archive\\tiny-imagenet-200\\test\\images"
map_path = os.path.join(os.path.dirname(image_folder), "label_to_image_map.json")

# Folder de ieșire pentru salvarea imaginilor rezultate
output_folder = "search_results"
os.makedirs(output_folder, exist_ok=True)  # Creează folderul dacă nu există

# Initializează clientul
client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Funcție pentru extragerea vectorului caracteristic
def extract_vector(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(64, 64, 3))
    vector = model.predict(img_array)
    return vector.flatten()

# Calea imaginii de căutat
image_path = "D:\\daci\\tbdproject\\archive\\tiny-imagenet-200\\test\\images\\test_3337.JPEG"

# Salvare imaginea de intrare în folderul de ieșire
input_image_name = os.path.basename(image_path)  # Obține doar numele fișierului
shutil.copy(image_path, os.path.join(output_folder, f"input_{input_image_name}"))
print(f"Imaginea de intrare a fost salvată ca: {os.path.join(output_folder, f'input_{input_image_name}')}")

# Încărcare mapare JSON
if not os.path.exists(map_path):
    raise FileNotFoundError(f"Fișierul JSON {map_path} nu există!")
with open(map_path, "r") as f:
    label_to_image_map = json.load(f)

# Validare imagini mapate
missing_files = []
for label, image_file in label_to_image_map.items():
    img_path = os.path.join(image_folder, image_file)
    if not os.path.exists(img_path):
        missing_files.append(img_path)

if missing_files:
    print("Următoarele fișiere lipsesc din folderul images:")
    for file in missing_files:
        print(file)
else:
    print("Toate fișierele de imagini sunt prezente conform mapării JSON.")

# Extragere vector caracteristic
query_vector = extract_vector(image_path).astype(np.float32)

# Căutare în Zilliz
# Măsurarea timpului pentru o căutare ANN
start_time = time.time()
search_result = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    metric_type="L2",
    params={"nprobe": 10},
    limit=15,
    output_fields=["label"]
)
end_time = time.time()

print(f"Timpul de căutare: {end_time - start_time:.4f} secunde")

# Debugging: Afișează rezultatele brute
print("Rezultatele brute din Zilliz:")
print(search_result)

# Verificare și salvare rezultate
print("Rezultatele căutării de imagine:")
for results in search_result:
    for i, result in enumerate(results):
        # Extrage label-ul din sub-câmpul 'entity'
        label = result.get("entity", {}).get("label", "N/A")
        print(f"Rezultatul {i + 1}: ID={result['id']}, Similaritate={result['distance']:.4f}, Label={label}")

        # Verificare dacă label-ul există în JSON
        if label in label_to_image_map:
            img_file = label_to_image_map[label]
            img_path = os.path.join(image_folder, img_file)
            if os.path.exists(img_path):
                # Salvare imagine
                result_image_name = f"result_{img_file}"
                output_path = os.path.join(output_folder, result_image_name)
                shutil.copy(img_path, output_path)
                print(f"Imaginea a fost salvată: {output_path}")
        else:
            print(f"Eroare: Label {label} nu există în JSON.")
