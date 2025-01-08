from pymilvus import MilvusClient, DataType
import numpy as np
import time
from dotenv import load_dotenv



load_dotenv()  # Încarcă variabilele din fișierul .env

CLUSTER_ENDPOINT = os.getenv("CLUSTER_ENDPOINT")
TOKEN = os.getenv("TOKEN")

if not CLUSTER_ENDPOINT or not TOKEN:
    raise ValueError("Variabilele din fișierul .env nu sunt setate corect!")


# Nume pentru colecție
COLLECTION_NAME = "tiny_imagenet_vectors"


# Initializează clientul Milvus
start_time = time.time()
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)
end_time = time.time()
print(f"Timpul de inițializare a clientului: {end_time - start_time:.4f} secunde")


# Creare schema pentru colecție
schema = MilvusClient.create_schema(
    auto_id=True,  # Cheia primară generată automat
    enable_dynamic_field=False  # Dezactivează câmpurile dinamice
)


# Adăugare câmpuri la schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)  # Primary key
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1280)  # Dimensiunea vectorului
schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=512)  # Eticheta (opțional)


# Creare colecție
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema
    )
    print(f"Colecția '{COLLECTION_NAME}' a fost creată cu succes!")
except Exception as e:
    print(f"Eroare la crearea colecției: {e}")
    exit()


# Creare parametri pentru indexare
try:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="L2"
    )


    # Creare index
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print(f"Indexul a fost creat cu succes pentru colecția '{COLLECTION_NAME}'!")
except Exception as e:
    print(f"Eroare la crearea indexului: {e}")
    exit()


# Încărcare colecție în memorie
try:
    client.load_collection(COLLECTION_NAME)
    print(f"Colecția '{COLLECTION_NAME}' a fost încărcată în memorie cu succes!")
except Exception as e:
    print(f"Eroare la încărcarea colecției: {e}")
    exit()


# Încărcare vectori procesați
feature_vectors = np.load("D:\\daci\\tbdproject\\archive\\tiny-imagenet-200\\test\\feature_vectors_mobilenet.npy")
labels = [f"label_{i}" for i in range(len(feature_vectors))]  # Exemplu de etichete


# Convertire vectori la float32
feature_vectors = feature_vectors.astype(np.float32)


# Pregătire date pentru inserare
entities = [
    {"vector": vector.tolist(), "label": label}
    for vector, label in zip(feature_vectors, labels)
]


# Inserare vectori în colecție
try:
    start_time = time.time()
    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=entities
    )
    end_time = time.time()  # Termină măsurarea
    print(f"Rezultatul inserării: {insert_result}")
    print(f"Timpul de inserare al vectorilor: {end_time - start_time:.4f} secunde")
   
except Exception as e:
    print(f"Eroare la inserarea datelor: {e}")
    exit()


# Verificare date inserate
try:
    collection_info = client.describe_collection(COLLECTION_NAME)
    print(f"Info colecție: {collection_info}")
except Exception as e:
    print(f"Eroare la descrierea colecției: {e}")


