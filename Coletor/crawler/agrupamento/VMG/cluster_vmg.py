import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import itertools
import matplotlib.pyplot as plt

YOUTUBERS_LIST = ['Julia MineGirl']


def gerar_vsmg_flatten_video(df_video: pd.DataFrame):
    """
    Gera a matriz de transição (VSMG) para um único vídeo a partir de um DataFrame.

    Parâmetros:
    - vetor_vsmg: pd.DataFrame
    Retorna:
    - numpy.ndarray -> VSMG unidimensional
    """

    # Verificação mínima
    for col in ['estado', 'proximo_estado', 'contagem']:
        if col not in df_video.columns:
            raise ValueError(f"O DataFrame deve conter a coluna '{col}'.")

    if df_video.empty:
        raise ValueError("O DataFrame fornecido está vazio.")

    somas_por_estado = df_video.groupby('estado')['contagem'].transform('sum')

    # Calcular probabilidade de transição
    df_video = df_video.copy()
    df_video['probabilidade'] = (df_video['contagem'] / somas_por_estado).fillna(0)

    vetor_vsmg = df_video['probabilidade'].to_numpy()
    
    return vetor_vsmg
    
def gerar_todos_vsmg_videos_youtuber(youtuber) -> np.ndarray: 
    # Criar um objeto Path para o diretório base do youtuber
    base_path = Path(f'files/{youtuber}')

    # Se o diretório do youtuber não existir, pular para o próximo
    if not base_path.is_dir():
        return None

    print(f'>>> Processando {youtuber}')

    primeira_vez_loop = True

    colunas_youtuber_df = ['video_id'] + [f'{i}-{j}' for i in range(1, 4) for j in range(1, 4)]
    linha_youtuber_df = []
    # .rglob('tiras_video.csv') busca recursivamente por este arquivo em todas as subpastas.
    for tiras_csv_path in base_path.rglob('transicoes_negatividade_3.csv'):
        video_data_path = tiras_csv_path.parent.parent


        transicoes_matriz = pd.read_csv(tiras_csv_path)
        vsmg_video = gerar_vsmg_flatten_video(transicoes_matriz)
        if primeira_vez_loop:
            todos_vsmg_youtuber = np.array(vsmg_video)
            primeira_vez_loop = False
        else:
            todos_vsmg_youtuber = np.vstack([todos_vsmg_youtuber,vsmg_video])

        video_df = pd.read_csv(f"{video_data_path}/videos_info.csv")
        video_id = video_df['video_id'].iloc[0]
        dados_linha = {'video_id': video_id}
        dados_linha.update({f'{i}-{j}': vsmg_video[(i - 1) * 3 + (j - 1)] for i in range(1, 4) for j in range(1, 4)})
        linha_youtuber_df.append(dados_linha)


    df_youtuber = pd.DataFrame(linha_youtuber_df, columns=colunas_youtuber_df)
    df_youtuber.to_csv(f"agrupamento/VMG/{youtuber}_vsmg_clusters.csv",index=False)
    print(f"Tamanho da Matriz ({youtuber}) 9x{todos_vsmg_youtuber.size/9}\n\n")

    return todos_vsmg_youtuber


def gerar_DBSCAN(matriz, youtuber, eps = 0.8, min_samples = 3):
    df_youtuber = pd.read_csv(f"agrupamento/VMG/{youtuber}_vsmg_clusters.csv")

    # Criar e ajustar o modelo DBSCAN

    # Ajuste 'eps' e 'min_samples' conforme necessário
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(matriz)

    # Mostrar resultados
    # print("Rótulos dos clusters:", labels)
    print("Número de clusters encontrados:", len(set(labels)) - (1 if -1 in labels else 0))
    print("Número de ruídos (label = -1):", np.sum(labels == -1))

    # Atualizar dados
    df_youtuber['Clusters'] = labels
    df_youtuber.to_csv(f"agrupamento/VMG/{youtuber}_vsmg_clusters.csv",index=False)

    # (Opcional) Visualizar os clusters em 2D com PCA
    X_2d = PCA(n_components=2).fit_transform(matriz)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='plasma', s=60)
    plt.title("Clusters DBSCAN (redução PCA)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.show()

    return labels

def gerar_DBSCAN_otimizado(X):
    print("Buscando melhores configurações (DBSCAN)...")

    # Parâmetros a testar
    eps_values = np.linspace(0.1, 2.0, 20)       # 20 valores entre 0.1 e 2.0
    min_samples_values = range(2, 10)             # de 2 a 9

    melhor_silhouette = -1
    melhor_config = None
    melhor_labels = None

    # Busca em grade
    for eps, min_samples in itertools.product(eps_values, min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Verifica se há pelo menos 2 clusters válidos
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue

        try:
            sil = silhouette_score(X, labels)
            if sil > melhor_silhouette:
                melhor_silhouette = sil
                melhor_config = (eps, min_samples)
                melhor_labels = labels
        except:
            continue

    print(f"✅ Melhor configuração encontrada:")
    print(f"eps = {melhor_config[0]:.2f}, min_samples = {melhor_config[1]}")
    print(f"Silhouette Score = {melhor_silhouette:.4f}")
    print(f"Número de clusters = {len(set(melhor_labels)) - (1 if -1 in melhor_labels else 0)}\n")

    return melhor_config


dados = gerar_todos_vsmg_videos_youtuber("Julia MineGirl")
eps, min_samples = gerar_DBSCAN_otimizado(dados)
gerar_DBSCAN(dados,eps=eps, min_samples=min_samples, youtuber = "Julia MineGirl")
# gerar_DBSCAN(dados,eps=0.5, min_samples=2)
