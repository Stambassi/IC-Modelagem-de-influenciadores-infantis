from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

'''
    Função para plotar o gráfico do Elbow Method
    @param features - DataFrame com os dados a serem testados
    @max_k - Número de clusters
'''
def plot_elbow_method(features: pd.DataFrame, max_k: int):
    inertia = []
    K = range(2, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-', label='Inertia')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.legend()

    # Salvar o gráfico em um arquivo
    plt.savefig("kmeans/graphs/elbow_method.png", dpi=300, bbox_inches='tight')  # DPI ajusta a resolução
    plt.close()  # Fecha a figura para evitar sobreposição em gráficos futuros

''' 
    Função para plotar o gáfico da pontuação de silhueta
    @param features - DataFrame com os dados a serem testados
    @max_k - Número de clusters
'''
def plot_silhouette_scores(features: pd.DataFrame, max_k: int):
    silhouette_scores = []
    K = range(2, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(features)
        silhouette_avg = silhouette_score(features, labels, metric='euclidean')
        silhouette_scores.append(silhouette_avg)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'ro-', label='Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.legend()

    # Salvar o gráfico em um arquivo
    plt.savefig("kmeans/graphs/silhouette_score.png", dpi=300, bbox_inches='tight')  # DPI ajusta a resolução
    plt.close()  # Fecha a figura para evitar sobreposição em gráficos futuros

''' 
    Função para plotar o gráfico dos clusters PCA
    @param features - DataFrame com os dados a serem testados
    @max_k - Número de clusters
'''
def plot_pca_clusters(features: pd.DataFrame, labels: np.ndarray):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', marker='.')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Clustered Data')
    plt.colorbar()

    # Salvar o gráfico em um arquivo
    plt.savefig("kmeans/graphs/pca_clusters.png", dpi=300, bbox_inches='tight')  # DPI ajusta a resolução
    plt.close()  # Fecha a figura para evitar sobreposição em gráficos futuros

'''
    Função para plotar o gráfico dos clusters com variáveis originais
    @param features - DataFrame com os dados a serem testados
    @param labels - Array com os rótulos de cluster gerados pelo KMeans
'''
def plot_original_clusters(features: pd.DataFrame, labels: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(features['duration'], features['comment_count'], c=labels, cmap='viridis', marker='.')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Comment count')
    plt.title('Clusters Based on Original Data')
    plt.colorbar(label='Cluster')
    
    # Salvar o gráfico em um arquivo
    plt.savefig("kmeans/graphs/original_clusters.png", dpi=300, bbox_inches='tight')  # DPI ajusta a resolução
    plt.close()  # Fecha a figura para evitar sobreposição em gráficos futuros

''' 
    Função para calcular Silhouette Score e Inertia do modelo treinado na base de dados
    @param features - DataFrame com os dados a serem testados
    @param n - Número de clusters a serem testados
    @return results - Lista com as
'''
def caluculate_metrics(features: pd.DataFrame, n: int) -> List[Dict[str, float]]:
    # Testar se a variável 'n' é inválida para esse contexto
    if n < 3: n = 3

    # Definir o intervalo de clusters a serem testados
    K = range(2, n)

    # Definir dicionário que armazenará as pontuações dos modelos
    results = []

    # Iniciar a repetição
    for k in K:
        # Criar a instância do KMeans
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')

        # Treinar o modelo e gerar os rótulos
        labels = kmeans.fit_predict(preprocessing.normalize(features))  # labels será um numpy.ndarray

        # Testar o silhouette_score e a inertia do modelo treinado
        silhouette_avg = silhouette_score(features, labels, metric='euclidean')
        inertia = kmeans.inertia_

        # Adicionar os resultados encontrado nas listas
        results.append({
            'k': k,
            'silhouette_score': silhouette_avg,
            'inertia': inertia
        })

    # Retornar
    return results

''''
    Função para selecionar o melhor número de clusters considerando ambos os critérios'
    @param result - Lista com a pontuação de cada cluster
    @return k = Número de clusters com a melhor pontuação
'''
def find_best_k(results: List[Dict[str, float]]) -> int:
    # Definir dados locais
    best_k = None
    melhor_pontuacao = float('-inf')
    
    for resultado in results:
        # Identificar valores do dicionário
        k = resultado['k']
        silhouette = resultado['silhouette_score']
        inertia = resultado['inertia']

        # Considerando a Silhouette Score como critério principal e penalizando levemente a Inertia
        pontuacao = silhouette - (inertia * 0.0001)

        # Testar a pontuação        
        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            best_k = k
    
    # Retornar
    return best_k

''''
    Função para tratar e normalizar os dados do DataFrame
    @param df - DataFrame a ser normalizado
    @param columns - Lista de colunas do DataFrame a serem normalizadas
    @return df_normalized_features - DataFrame normalizado
'''
def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    # Selecionar colunas relevantes para a clusterização
    df_features = df[columns]

    # Aplicar a padronização dos dados com média 0 e desvio padrão 1
    scaler = MinMaxScaler()
    df_normalized_features = scaler.fit_transform(df_features)

    # Retornar o DataFrame normalizado
    return df_normalized_features

''''
    Função para reduzir os dados do DataFrame, removendo os outliers
    @param df - DataFrame a ser reduzido
    @param columns - Lista de colunas do DataFrame a serem reduzidas
    @return df_reduced_features - DataFrame reduzido
'''
def reduce_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    # Selecionar colunas relevantes para a redução
    df_features = df[columns]

    # Calcular os percentis 25%, 50% e 75%
    Q1 = df_features.quantile(0.25)
    Q3 = df_features.quantile(0.75)

    # Calcular os limites inferior e superior
    lower_limit = Q1 - 1.5 * (Q3 - Q1)
    upper_limit = Q3 + 1.5 * (Q3 - Q1)


    # Filtrar os dados que estão entre os percentis
    df_reduced_features = df_features[(lower_limit < df_features) & (df_features < upper_limit)]

    df_features = df_features.dropna(subset=columns)

    # Retornar o DataFrame reduzido
    return df_reduced_features

''''
    Função controladora do algoritmo KMeans
    @param n - Número máximo de clusters a serem testados
    @param columns - Lista de colunas a serem agrupadas
    @param normalized - True para normalizar, False caso contrário
    @param reduced - True para remover outliers, False caso contrátio
    @param plot - True para plotar gráficos e salvar como arquivos de imagem, False caso contrário
    @param show_videos - True para mostrar exemplos de vídeos de cada cluster no terminal, False caso contrário
'''
def kmeans(n: int, columns: List[str], normalized: bool, reduced: bool, fixed: bool, plot: bool, show_videos: bool):
    # Definir dados
    df_principal = pd.DataFrame()

    # Testar quantidade de tentativas
    if n < 3: n = 3

    # Importar base de dados
    df = pd.read_csv('kmeans/kmeans_video.csv')
    df_principal = df

    # Remover as linhas que possuem valores inválidos
    df_principal = df_principal.dropna(subset=columns)

    # Testar se é para normalizar os dados
    if normalized:
        df_principal = normalize_data(df_principal, columns)
    else:
        df_principal = df[columns]

    # Testar se é para reduzir os dados
    if reduced:
        df_principal = reduce_data(df_principal, columns)

    # Testar se é para encontrar o número ideal de clusters
    if fixed:
        best_n = int(input("Digite a quantidade fixa de clusters: "))
    else:
        best_n = find_best_k( caluculate_metrics(df_principal, n) )

    # Aplicar o KMeans
    kmeans = KMeans(n_clusters=best_n, random_state=0, n_init='auto')
    kmeans.fit(df_principal)

    # Adicionar os rótulos de cluster ao DataFrame original
    df['cluster'] = kmeans.labels_

    # Exibir o número de clusters
    print("Número de clusters:", best_n)

    # Contar a quantidade de ocorrências de cada cluster
    cluster_counts = df['cluster'].value_counts()

    # Exibir a contagem de ocorrências de cada cluster
    print("Contagem de ocorrências por cluster:")
    print(cluster_counts)

    # Testar se é para mostrar os gráficos da quantidade de cluster
    if plot:
        plot_elbow_method(df_principal, 10)
        plot_silhouette_scores(df_principal, 10) 
        plot_pca_clusters(df_principal, kmeans.labels_)
        if len(columns) == 2:
            plot_original_clusters(df_principal, kmeans.labels_)

    # Obter os valores únicos da coluna 'cluster'
    unique_clusters = df['cluster'].unique()

    # Analisar as características principais de cada cluster
    for cluster in unique_clusters:
        print(f'\nCluster {cluster}:')
        cluster_data = df[df['cluster'] == cluster]
        print(cluster_data.describe().loc[['mean', 'std'], columns])
        # Exibir alguns exemplos de comentários no cluster
        if show_videos:
            print("\nExemplos de vídeos:")
            print(cluster_data[['video_id', 'duration', 'comment_count']].head(10))

    # Salvar dados em arquivo separado
    df.to_csv("kmeans/kmeans_video_clustered.csv", index=False)


if __name__ == "__main__":
    # Definir as colunas de interesse
    columns = ['duration', 'comment_count']
    # columns = ['duration', 'comment_count', 'view_count']
    # columns = ['duration', 'comment_count', 'like_count']
    # columns = ['duration', 'comment_count', 'like_count', 'view_count']

    # Chamar a execução
    kmeans(n=10, columns=columns, normalized=False, reduced=True, fixed=False, plot=False, show_videos=False)