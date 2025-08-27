import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

'''
    Função para gerar um DataFrame com todos os vídeos e salvar em um arquivo csv
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def salvar_toxicidade(youtubers_list: list[str]) -> None:
    # Criar DataFrame
    df_dados_videos = pd.DataFrame(columns=['video_id', 'cluster'])
    # Percorrer a lista de youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            print(f'Aviso: Diretório para {youtuber} não encontrado. Pulando.')
            continue

        print(f'>>> Processando {youtuber}')
        
        # .rglob() busca recursivamente pelo arquivo em todas as subpastas
        arquivos_encontrados = list(base_path.rglob('dados_percentis_normalizados.csv'))
        
        if not arquivos_encontrados:
            print(f'Nenhum arquivo "dados_percentis_normalizados.csv" encontrado para {youtuber}.')
            continue
            
        for percentis_normalizados_csv_path in arquivos_encontrados:
            try:                
                df_percentis_normalizados = pd.read_csv(percentis_normalizados_csv_path)

                # Garantir que a coluna esperada existe no arquivo
                if 'toxicidade_normalizada' not in df_percentis_normalizados.columns:
                    print(f'Aviso: Coluna "toxicidade_normalizada" não encontrada em {percentis_normalizados_csv_path}. Pulando arquivo.')
                    continue

                # Encontrar a pasta do vídeo
                video_path = percentis_normalizados_csv_path.parent
                
                # Encontrar arquivo com informações do vídeo
                videos_info_csv_path = video_path / 'videos_info.csv'

                try:                    
                    # Encontrar o identificador do vídeo
                    df_videos_info = pd.read_csv(videos_info_csv_path)
                    video_id = df_videos_info['video_id'].iloc[0]

                    # Criar informações da nova linha
                    dict_video = {
                        'video_id': video_id,
                    }

                    # Adicionar ao DataFrame geral
                    df_dados_videos.loc[ len(df_dados_videos) ] = dict_video
                except Exception as e:
                    print(f'Erro ao processar o arquivo {videos_info_csv_path}: {e}')
            except Exception as e:
                print(f'Erro ao processar o arquivo {percentis_normalizados_csv_path}: {e}')

        print(f'Os dados do {youtuber} foram coletados.')

    if len(df_dados_videos) > 0:
        base_path = Path('agrupamento/toxicidade')
        caminho_salvar = base_path / 'percentis.csv'
        df_dados_videos.to_csv(caminho_salvar, index=False)

        print(f'Percentis foram salvos no caminho {caminho_salvar}')
    else:
        print(f'Nenhum dado foi encontrado')

'''
    Função para buscar todos os dados de toxicidade divididos em percentis
    @param youtubers_list - Lista de youtubers a serem analisados
    @return dados_videos - Matriz NumPy com todos os dados de toxicidade
'''
def toxicidade_percentis(youtubers_list: list[str]) -> np.ndarray:
    # Iniciar com uma lista Python vazia.
    videos_data_list = []

    # Percorrer a lista de youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            print(f'Aviso: Diretório para {youtuber} não encontrado. Pulando.')
            continue

        print(f'>>> Processando {youtuber}')
        
        # .rglob() busca recursivamente pelo arquivo em todas as subpastas
        arquivos_encontrados = list(base_path.rglob('dados_percentis_normalizados.csv'))
        
        if not arquivos_encontrados:
            print(f'Nenhum arquivo "dados_percentis_normalizados.csv" encontrado para {youtuber}.')
            continue
            
        for percentis_normalizados_csv_path in arquivos_encontrados:
            try:                
                df_percentis_normalizados = pd.read_csv(percentis_normalizados_csv_path)

                # Garantir que a coluna esperada existe no arquivo
                if 'toxicidade_normalizada' not in df_percentis_normalizados.columns:
                    print(f'Aviso: Coluna "toxicidade_normalizada" não encontrada em {percentis_normalizados_csv_path}. Pulando arquivo.')
                    continue

                percentis_normalizados = df_percentis_normalizados['toxicidade_normalizada'].tolist()

                # Adicionar a lista de toxicidade à lista Python.
                if percentis_normalizados:
                    videos_data_list.append(percentis_normalizados)
                else:
                    print(f'Aviso: Nenhum dado de toxicidade encontrado em {percentis_normalizados_csv_path}.')

            except Exception as e:
                print(f'Erro ao processar o arquivo {percentis_normalizados_csv_path}: {e}')

        print(f'Os dados do {youtuber} foram coletados.')

    # Converter a lista de listas em um array NumPy de 2D no final
    if videos_data_list:
        try:
            # Verificar se todos os vetores têm o mesmo comprimento
            tamanho_primeiro_vetor = len(videos_data_list[0])
            if not all(len(v) == tamanho_primeiro_vetor for v in videos_data_list):
                print("Erro Crítico: Os vetores de percentis têm tamanhos inconsistentes. Não é possível criar a matriz.")
                return np.array([]) # Retorna um array vazio

            dados_videos = np.array(videos_data_list)
            print(f'\nColeta finalizada. Matriz de dados criada com sucesso no formato: {dados_videos.shape}')
            return dados_videos
        except Exception as e:
            print(f"Erro ao converter a lista para array NumPy: {e}")
            return np.array([]) # Retorna um array vazio
    else:
        print('\nNenhum dado foi coletado. Retornando um array vazio.')
        return np.array([]) # Retorna um array vazio

'''
    Função para encontrar o número ideal de clusters (k) com o Método do Cotovelo (Elbow Method)
    @param dados_videos - Matriz com todos os dados de toxicidade
'''
def plot_elbow_method(dados_videos: np.ndarray) -> None:
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

    base_path = Path('agrupamento/toxicidade')
    pasta_graficos = base_path / 'graficos'
    pasta_graficos.mkdir(parents=True, exist_ok=True)    
    caminho_salvar = pasta_graficos / 'elbow_method_percentis.png'
    plt.savefig(caminho_salvar, dpi=150)
    plt.close()
    print(f"Gráfico do Elbow Method salvo com sucesso em: {caminho_salvar}")

'''
    Função para encontrar o número ideal de clusters (k) com o Silhouette Score
    @param dados_videos - Matriz com todos os dados de toxicidade
'''
def plot_silhouette_score(dados_videos: np.ndarray) -> None:
    # O score de silhueta precisa de no mínimo 2 clusters
    range_n_clusters = range(2, 11) 
    silhouette_avg = []

    print("Calculando Silhouette Score para diferentes valores de k...")

    for n_clusters in range_n_clusters:
        # Inicializar o KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(dados_videos)

        # Calcular a média do silhouette score
        score = silhouette_score(dados_videos, cluster_labels)
        silhouette_avg.append(score)
        print(f"Para n_clusters = {n_clusters}, o Silhouette Score médio é : {score:.4f}")

    # Plotar o gráfico do Silhouette Score
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, silhouette_avg)
    plt.title('Análise de Silhueta para Encontrar o k Ideal')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette Score Médio')
    plt.grid(True)

    base_path = Path('agrupamento/toxicidade')
    pasta_graficos = base_path / 'graficos'
    pasta_graficos.mkdir(parents=True, exist_ok=True)    
    caminho_salvar = pasta_graficos / 'silhouette_score_percentis.png'

    plt.savefig(caminho_salvar, dpi=150)
    plt.close()
    print(f"Gráfico do Silhouette Score salvo com sucesso em: {caminho_salvar}")

    # O melhor k é aquele com o maior score
    k_ideal = range_n_clusters[np.argmax(silhouette_avg)]
    print(f"\nO melhor número de clusters (k) segundo a Análise de Silhueta é: {k_ideal}")

'''
    Treinar o modelo com o k-ideal encontrado
    @param dados_videos - Matriz com todos os dados de toxicidade
    @param k_ideal - Número ideal de clusters
'''
def fit_model(dados_videos: np.ndarray, k_ideal: int) -> None:
    # Criar e treinar o modelo final
    kmeans_final = KMeans(n_clusters=k_ideal, n_init=10, random_state=42)
    cluster_labels = kmeans_final.fit_predict(dados_videos)

    # Salvar os resultados
    percentis_csv_path = 'agrupamento/toxicidade/percentis.csv'
    try:
        df_percentis = pd.read_csv(percentis_csv_path)
        df_percentis['cluster'] = cluster_labels

        base_path = Path('agrupamento/toxicidade')
        caminho_salvar = base_path / 'percentis.csv'
        df_percentis.to_csv(caminho_salvar, index=False)

        print(f'Clusters dos percentis foram salvos no caminho {caminho_salvar}')
    except Exception as e:
        print(f'Inválido ao processar arquivo em {percentis_csv_path}: {e}')

'''
    Função para comparar graficamente os clusters de vídeos 
    @param dados_csv_file_name - Nome do arquivo csv com os dados de toxicidade de cada vídeo   
    @param clusters_csv_file_name - Nome do arquivo csv com os clusters de cada vídeo
    @param n_amostras_por_cluster - Número de vídeos de exemplo a serem mostrados por cluster.
'''
def compare_clusters(dados_csv_file_name: str = 'dados_percentis_normalizados.csv', 
                    clusters_csv_file_name: str = "agrupamento/toxicidade/percentis.csv", 
                    n_amostras_por_cluster: int = 5) -> None:
    # Encontrar os dados de todos os vídeos
    print("Mapeando todos os arquivos de dados de percentis...")
    base_dir = Path('files/')
    todos_os_arquivos_de_dados = list(base_dir.rglob(dados_csv_file_name))
    
    # Dicionário com o par (identificador do vídeo, caminho para a pasta de dados do vídeo)
    video_id_to_path_map = {}
    for path in todos_os_arquivos_de_dados:
        # Encontrar a pasta do vídeo
        video_dir = path.parent
        video_info = video_dir / 'videos_info.csv'

        try:
            # Encontrar o id do vídeo
            df_video_info = pd.read_csv(video_info)
            video_id = df_video_info['video_id'].iloc[0]
        
            # Adicionar o arquivo de dados
            video_id_to_path_map[video_id] = path
        except Exception as e:
            print(f'Inválido ao processar arquivo em {video_info}: {e}')
    
    if not video_id_to_path_map:
        print(f"Erro: Nenhum arquivo '{clusters_csv_file_name}' encontrado na pasta 'files/'.")
        return

    # Carregar as informações de cluster
    try:
        df_clusters = pd.read_csv(clusters_csv_file_name)
    except Exception as e:
        print(f"Inválido ao processar o arquivo {clusters_csv_file_name}: {e}")
        return

    # Criar uma lista com o número de cada cluster
    clusters_unicos = sorted(df_clusters['cluster'].unique())
    num_clusters = len(clusters_unicos)

    print(f"Análise de {num_clusters} clusters encontrados. Mostrando {n_amostras_por_cluster} exemplos por cluster.")

    # Criar a estrutura do Gráfico (Facet Grid
    fig, axes = plt.subplots(
        nrows=n_amostras_por_cluster, 
        ncols=num_clusters, 
        figsize=(4 * num_clusters, 2.0 * n_amostras_por_cluster)
    )

    # Garantir que 'axes' seja sempre uma matriz 2D para facilitar a indexação
    if n_amostras_por_cluster == 1:
        axes = np.array(axes).reshape(1, -1)
    if num_clusters == 1:
        axes = np.array(axes).reshape(-1, 1)

    # Variável para guardar o objeto do heatmap para a colorbar
    im = None 

    # Iterar por cada coluna (cluster) e plotar seus vídeos de amostra
    for col_idx, cluster_id in enumerate(clusters_unicos):
        # Filtrar os vídeos que pertencem ao cluster atual
        videos_no_cluster = df_clusters[df_clusters['cluster'] == cluster_id]['video_id'].tolist()
        
        # Selecionar uma amostra aleatória de vídeos
        num_a_amostrar = min(len(videos_no_cluster), n_amostras_por_cluster)
        videos_amostrados = np.random.choice(videos_no_cluster, size=num_a_amostrar, replace=False)

        # Adicionar um título para a coluna do cluster
        axes[0, col_idx].set_title(f'Cluster {cluster_id}\n({len(videos_no_cluster)} vídeos)', fontsize=14, pad=20)
        
        # Iterar pelas linhas (vídeos) para plotar cada vídeo da amostra
        for row_idx in range(n_amostras_por_cluster):
            ax = axes[row_idx, col_idx]
            
            # Se tivermos um vídeo para esta linha, plote-o
            if row_idx < len(videos_amostrados):
                video_id = videos_amostrados[row_idx]
                
                try:
                    # Carregar os dados de toxicidade do vídeo
                    path_do_video = video_id_to_path_map[video_id]
                    df_toxicidade = pd.read_csv(path_do_video)
                    coordenadas_percentis = df_toxicidade['toxicidade_normalizada'].tolist()

                    # Lógica de plotagem adaptada da sua função
                    dados_heatmap = np.array(coordenadas_percentis).reshape(1, -1)
                    im = ax.imshow(dados_heatmap, cmap='Reds', vmin=0, vmax=1.0, aspect='auto')

                    # Tornar o identificador do vídeo o título do subplot
                    ax.set_title(video_id, fontsize=9, pad=5)

                    # Esconder o eixo y
                    ax.get_yaxis().set_visible(False)
                    
                    # Mostrar labels do eixo X apenas na última linha de gráficos
                    if row_idx == n_amostras_por_cluster - 1:
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
                    else:
                        ax.get_xaxis().set_visible(False)

                except (KeyError, FileNotFoundError):
                    ax.text(0.5, 0.5, f"Dados não\nencontrados\npara {video_id}", ha='center', va='center')
                    ax.set_visible(False) # Esconde o eixo se houver erro
            else:
                # Se não houver mais vídeos para amostrar neste cluster, esconde o eixo
                ax.set_visible(False)

    # Limpeza e finalização do gráfico
    fig.suptitle('Comparação Visual dos Clusters de Toxicidade de Vídeos', fontsize=20)
    fig.supxlabel('Progresso Percentual da Duração do Vídeo', fontsize=14, y=0.03)

    if im:
        # Criar um eixo dedicado para a legenda para controle total da posição.
        cbar_ax = fig.add_axes([0.25, 0.88, 0.5, 0.025]) # [25% da esq, 88% de baixo, 50% de larg, 2.5% de alt]

        cbar = fig.colorbar(
            im, 
            cax=cbar_ax,                 # Usa o eixo dedicado que acabamos de criar
            orientation='horizontal'     # Define a orientação
        )
        cbar.set_label('Nível de Toxicidade Normalizado', fontsize='12')

    # Ajustar de layout para criar o espaçamento para a legenda
    fig.subplots_adjust(
        left=0.05,      # Padding esquerdo
        right=0.95,     # Padding direito
        top=0.84,       # Espaço para o super-título E para a legenda acima dos gráficos
        bottom=0.12,    # Espaço para o sub-título e eixos X
        hspace=0.6,     # Espaço vertical entre os gráficos
        wspace=0.2      # Espaço horizontal entre os gráficos
    )

    # Salvar o gráfico
    pasta_graficos = Path('agrupamento/toxicidade/graficos')
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    caminho_salvar = pasta_graficos / 'comparacao_clusters_heatmap.png'
    plt.savefig(caminho_salvar, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nGráfico de comparação de clusters salvo em: {caminho_salvar}")



if __name__ == '__main__':
    lista_youtubers =  ['AuthenticGames', 'Cadres']
    k_ideal = 2

    #salvar_toxicidade(lista_youtubers)

    #dados_videos = toxicidade_percentis(lista_youtubers)

    #plot_elbow_method(dados_videos)
    #plot_silhouette_score(dados_videos)

    #fit_model(dados_videos, k_ideal)

    compare_clusters()
