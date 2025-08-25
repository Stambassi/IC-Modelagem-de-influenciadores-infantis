import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def rodar_kmeans(dados_videos: np.ndarray) -> None:
    # Encontrar o número ideal de clusters (k) com o Método do Cotovelo (Elbow Method)
    wcss = [] # Within-Cluster Sum of Squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(dados_videos)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss)
    plt.title('Método do Cotovelo para encontrar o k ideal')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # --- Passo 2: Aplicar o K-Means com o k escolhido ---
    # Supondo que o cotovelo aponte para, digamos, k=4
    k_ideal = 4
    kmeans = KMeans(n_clusters=k_ideal, init='k-means++', max_iter=300, n_init=10, random_state=42)
    predicoes_cluster = kmeans.fit_predict(dados_videos)

    # 'predicoes_cluster' agora é um array onde cada elemento é o ID do cluster 
    # para o vídeo correspondente. Ex: [0, 2, 1, 0, 3, ...]

    # --- Passo 3: Analisar os resultados ---
    print("Centróides dos clusters:")
    print(kmeans.cluster_centers_)