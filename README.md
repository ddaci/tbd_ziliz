# tbd_ziliz

Sistem de căutare a imaginilor similare utilizând Zilliz - bază de date specializată pentru embeddings într-un mediu cloud-native.

##  **Prezentare generală**
Gestionarea și procesarea datelor nestructurate ridică provocări tehnice precum:
- **Indexare și căutare eficientă**
- **Scalabilitate și performanță**

Adoptarea bazelor de date pentru embeddings în mediile cloud-native asigură:
- Gestionarea eficientă a datelor nestructurate
- Răspunsuri rapide și precise în timpul căutărilor
- Flexibilitate și scalabilitate în infrastructuri moderne

##  Arhitectură
Platforma utilizează **Zilliz Cloud**, un serviciu complet gestionat bazat pe **Milvus**, o bază de date vectorială open-source.

### Pipeline de procesare

1. Selecția setului de date:
   Dataset-ul **Tiny ImageNet-200** (10.000 de imagini) este utilizat pentru testare.  
   👉 [Descărcare dataset](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200/data)

   **Limitările planului gratuit Zilliz Cloud:**
   - 5 colecții (tabele)
   - 5 GB stocare (≈ 1 milion vectori de 768 dimensiuni)
   - 2,5 milioane vCU/lună pentru căutări și inserări

2. Extracția embeddings:
   Modelul pre-antrenat **MobileNetV2** extrage vectori de lungime **1280** din imagini, optim pentru dimensiuni mici.

3. Stocarea embeddings: 
   Vectorii sunt salvați în baza de date **Milvus** găzduită în cloud.

4. Interogarea bazei de date:
   O imagine încărcată este transformată într-un vector embedding, care inițiază o căutare **Approximate Nearest Neighbor (ANN)** pentru a returna cele mai similare imagini (Top-K).

### Optimizarea căutării
Parametrii cheie pentru optimizare includ:
- **Nprobe:** Controlează precizia căutării ANN *(ideal: 5-20)*  
- **Metrici:** Se poate experimenta cu **L2**, **IP**, etc.

Exemplu de cod pentru interogare:
```python
search_result = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    metric_type="L2",
    params={"nprobe": 10},
    limit=5,
    output_fields=["label"]
)
```
### Măsurarea performanței
Metadate Zilliz Cloud:
- **Search Latency**: Timpul de răspuns la căutare
- **Insert Time**: Timpul de inserare a vectorilor
- **vCPU Usage**: Resurse utilizate pentru operațiuni
- **Stocare**: Spațiu ocupat
Metrici în cod:
- **Timp de căutare**: Performanța căutărilor ANN
- **Timp de inserare**: Durata adăugării vectorilor

**Rezultate:**

Timp de căutare(performanța căutărilor ANN):
-  0.6882 secunde
-  0.2071 secunde
-  0.2129 secunde
-  0.6772 secunde
-  0.6876 secunde
-  0.6816 secunde

Timpul de inserare al vectorilor:
    Durata efectivă necesară pentru a adăuga vectorii embeddings în colecția din Zilliz a fost: 14.3972 secunde pentru 10000 de vectori de dimensiune 1280

Timpul de inițializare a clientului: 0.3125 secunde


### Organizarea codului
Proiectul rulează local în Python, conectându-se la baza de date Milvus din cloud.

![schematbdziliz](https://github.com/user-attachments/assets/d423f50c-e1fe-4ba6-9d07-069640004da6)

### Verificam vectorii extrasi:

### 1. Dimensiunea vectorilor
   - Shape of feature vectors: (10000, 1280)
   - În total, există 10.000 de vectori, corespunzători celor 10.000 de imagini din setul de date.
   - Fiecare vector are 1.280 de dimensiuni, ceea ce corespunde arhitecturii modelului MobileNetV2 în starea sa pre-antrenată.

     Acesta returnează un vector de caracteristici cu 1.280 de elemente pentru fiecare imagine.

     
### 2. Verificarea vectorilor extrași
   - First feature vector: [0. 0. 0. ... 0. 0. 0.]
   - Primul vector este afișat, iar toți elementele acestuia sunt 0.
   - All vectors zero: False indică faptul că nu toți vectorii sunt nuli(vectorii nuli indică faptul că imaginea nu conține informații distinctive pentru modelul folosit, MobileNetV2).

     
### 3. Valori minime și maxime
   - Minimum value in vectors: 0.0
   - Valoarea minimă dintre toate elementele vectorilor este 0.0.

     Acest lucru este de așteptat, deoarece MobileNetV2 normalizează și procesează imaginile astfel încât multe dintre elementele vectorilor să fie nule pentru regiuni neinformative.
   - Maximum value in vectors: 6.0
   - Valoarea maximă găsită este 6.0, ceea ce indică faptul că unele elemente ale vectorilor au activări puternice.

     Aceasta semnalează regiuni distinctive în imaginile procesate.

Codul din insert_vect_ziliz_2.py definește și implementează pașii pentru crearea unei colecții în Milvus/Zilliz Cloud, configurarea indexării și inserarea vectorilor embeddings generați din imagini.

### 1. Configurarea conexiunii cu Zilliz Cloud

Se initializează un client Milvus folosind endpoint-ul și token-ul API furnizate.

Conexiunea permite interacțiunea cu serviciul cloud pentru a crea, administra și utiliza colecții.

### 2. Crearea unei colecții în baza de date vectorială

O schemă este definită pentru colecție, incluzând următoarele câmpuri:
      - id: Cheia primară (generată automat).
      - vector: Vectorii embeddings extrași din imagini (tip FLOAT_VECTOR cu dimensiunea 1280).
      - label: Etichetele asociate imaginilor (tip VARCHAR cu o lungime maximă de 512 caractere).

Colecția este creată în Zilliz Cloud utilizând schema definită.

### 3. Configurarea indexării pentru căutări eficiente

Indexarea este configurată pe baza câmpului vector:

Tipul indexului: AUTOINDEX, o metodă care ajustează automat parametrii pentru performanță.

Metoda de măsurare a similitudinii: L2 (distanța euclidiană).

Indexul este creat pentru a permite căutări rapide în spațiul vectorial.
### 4. Încărcarea colecției în memorie

Colecția este încărcată în memorie pentru a permite operațiuni rapide de inserare și căutare.

### 5. Inserarea vectorilor embeddings și a etichetelor

Vectorii embeddings preprocesați (stocați în fișierul feature_vectors_mobilenet.npy) sunt încărcați și convertiți în format float32.

Etichetele asociate sunt generate automat (label_0, label_1, etc.).

Datele sunt pregătite sub formă de entități.

Datele sunt inserate în colecție folosind funcția insert.

Masuram:

Timpul de inserare al vectorilor: Durata efectivă necesară pentru a adăuga vectorii embeddings în colecția din Zilliz.14.3972 secunde pentru 10000 de vectori de dimensiune 1280

Timpul de inițializare a clientului: 0.3125 secunde

### Cum arată baza de date vectorială în Ziliz Cloud

![zilizcl](https://github.com/user-attachments/assets/3685192f-061f-41d3-b95d-2b257858665d)


### Rezultate
![rezultstbd](https://github.com/user-attachments/assets/b94aca5a-e060-44ff-830a-202141dbb232)


