import numpy as np

# Încarcă vectorii din fișier
vectors = np.load(r"D:\daci\tbdproject\archive\tiny-imagenet-200\test\feature_vectors_mobilenet.npy")

# Afișează dimensiunea și primul vector pentru verificare
print("Shape of feature vectors:", vectors.shape)
print("First feature vector:", vectors[0])
print("All vectors zero:", np.all(vectors == 0))  # Ar trebui să returneze False
print("Minimum value in vectors:", np.min(vectors))
print("Maximum value in vectors:", np.max(vectors))
for i in range(3):  # Afișează primii 3 vectori
    print(f"Vector {i}:", vectors[i])
