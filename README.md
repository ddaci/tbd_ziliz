# tbd_ziliz
Sistem de căutare a imaginilor similare utilizând Ziliz - bază de date specializată pentru embeddings într-un mediu cloud-native.

Arhitectura:
Zilliz Cloud, o platformă complet gestionată care oferă servicii de baze de date vectoriale în cloud, bazată pe Milvus - bază de date vectorială open-source. 

Selecția setului de date: Utilizăm un set de imagini deja existente: tiny-imagenet-200(contine 10000 de imagini). Descarcam datasetul de aici: https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200/data

La selecția datelor, am ținut cont de limitările impuse de planul gratuit Zilliz Cloud.

Extracția embeddings:
Imagini: Utilizăm MobileNetV2 pre-antrenat pentru a obține vectori embeddings de lungime 1280. Modelul este potrivit pentru setul nostru de date cu imagini de dimensiuni foarte mici.

Stocarea embeddings: Vectorii sunt stocați într-o bază de date vectorială Milvus găzduită în Cloud.

Interogarea bazei de date:
Utilizatorul trimite o imagine către aplicație, care o transformă într-un vector embedding. Acest vector este utilizat pentru a iniția o căutare Approximate Nearest Neighbor (ANN) în baza de date vectorială, identificând cele mai similare rezultate din setul de date.
Cele mai relevante imagini (Top-K) sunt returnate utilizatorului.

Configurarea parametrilor pentru optimizare
Optimizarea căutării:


search_result = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    metric_type="L2",
    params={"nprobe": 10},
    limit=5,
    output_fields=["label"]
)

Nprobe: Parametru pentru precizia căutării ANN. Valori mai mici reduc latența dar pot afecta precizia. Încearcăm valori între 5 și 20 pentru echilibru.
Putem experimenta cu diferite metrici (L2, IP, etc.).

Măsurarea performanței: 
In cod In cod am măsurat:
Timp de căutare(performanța căutărilor ANN)
Timpul de căutare: 0.6882 secunde
Timpul de căutare: 0.2071 secunde
Timpul de căutare: 0.2129 secunde
Timpul de căutare: 0.6772 secunde
Timpul de căutare: 0.6876 secunde
Timpul de căutare: 0.6816 secunde
Timpul de inserare al vectorilor: Durata efectivă necesară pentru a adăuga vectorii embeddings în colecția din Zilliz a fost: 14.3972 secunde pentru 10000 de vectori de dimensiune 1280
Timpul de inițializare a clientului: 0.3125 secunde
In Ziliz, baza de date vectoriala ce contine vectorii a 10.000 de imagini de 64x64 pixeli ocupa 0.7 GB.

Codul in Python ruleaza local si se conecteaza la baza de date vectoriala Milvus in cloud. 
