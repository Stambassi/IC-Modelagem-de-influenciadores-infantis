import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype
import numpy as np

import itertools
import matplotlib.pyplot as plt
from rich.console import Console

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

console = Console()

# Define as regras para cada tipo de análise (métrica)
METRICAS_CONFIG = {
    'sentimento': {
        'coluna_base': 'sentimento_dominante', # Coluna no 'tiras_video.csv' que define o estado
        'tipo_estados': 'categorico', # 'categorico' (ex: POS, NEU) ou 'numerico' (ex: 0.0-1.0)
        'estados': ['POS', 'NEU', 'NEG'] # Lista de estados para análises categóricas
    },
    'negatividade': {
        'coluna_base': 'negatividade',
        'tipo_estados': 'numerico',
        'n_estados': 3 # Número de 'bins' para dividir o score (ex: 3 estados)
    },
    'toxicidade': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', # Novo tipo de estado
        'limiares': [0.0, 0.30, 0.70, 1.01], # 1.01 para garantir que 1.0 seja incluído
        'estados': ['NT', 'GZ', 'T'] # Nomes dos estados
    }
}

'''
    Função para gerar a matriz de transição (VMG) para um único vídeo e achatá-la para um vetor
    @param df_transicoes - DataFrame com as contagens de transição
    @param estados - Lista com os nomes/rótulos dos estados (ex: [1, 2, 3] ou ['POS', 'NEU', 'NEG'])
    @return numpy.ndarray - O VMG como um vetor unidimensional (N*N elementos)
'''
def gerar_vmg_flatten_video(df_transicoes: pd.DataFrame, estados: list) -> np.ndarray:
    # Calcular as probabilidades de transição
    somas_por_estado = df_transicoes.groupby('estado')['contagem'].transform('sum')
    df_transicoes = df_transicoes.copy()
    df_transicoes['probabilidade'] = (df_transicoes['contagem'] / somas_por_estado).fillna(0)

    # Criar os rótulos de estado (ex: [1, 2, 3] ou ['POS', 'NEU', 'NEG'])
    labels = estados
    
    # Garantir que as colunas sejam do tipo Categórico para o pivot
    tipo_categorico = CategoricalDtype(categories=labels, ordered=True)
    df_transicoes['estado'] = df_transicoes['estado'].astype(tipo_categorico)
    df_transicoes['proximo_estado'] = df_transicoes['proximo_estado'].astype(tipo_categorico)

    # Pivotar os dados para o formato de matriz
    matriz = df_transicoes.pivot(index='estado', columns='proximo_estado', values='probabilidade')
    
    # Reindexar a matriz para garantir que ela seja N x N
    matriz = matriz.reindex(index=labels, columns=labels, fill_value=0.0)
    
    # Achatar (flatten) a matriz para um vetor 1D
    vetor_vmg = matriz.to_numpy().flatten()
    
    return vetor_vmg

'''
    Função para processar todos os vídeos de um youtuber, gerar o VMG de cada um e salvar os dados
    @param youtuber - Nome do youtuber a ser processado
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3')
    @param metrica_config - Dicionário de configuração da métrica
    @return tuple - Contendo a matriz de VMGs (vídeos x features), o DataFrame e o caminho do arquivo salvo
'''
def preparar_dados_agrupamento(youtuber: str, nome_analise: str, metrica_config: dict) -> tuple: 
    console.print(f'>>> Processando VMGs para [cyan]{youtuber}[/cyan] (Análise: {nome_analise})')
    
    # Criar um objeto Path para o diretório base do youtuber
    base_path = Path(f'files/{youtuber}')

    if not base_path.is_dir():
        console.print(f"[red]Diretório não encontrado para {youtuber}. Pulando.[/red]")
        return None, None, None

    # Salvar os arquivos na pasta 'agrupamento' de cada youtuber
    output_dir = base_path / 'agrupamento'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo de saída agora inclui o nome da análise
    output_csv_path = output_dir / f'vmg_cluster_data_{nome_analise}.csv'

    # Coletar VMG de cada vídeo
    lista_vmg_videos = [] # Lista para os dados numpy (para o clustering)
    lista_info_videos = [] # Lista para os dados do DataFrame (para o CSV)
    
    # Define os estados e o nome do arquivo de transições dinamicamente
    if metrica_config['tipo_estados'] == 'categorico' or metrica_config['tipo_estados'] == 'numerico_categorizado':
        estados = metrica_config['estados']
    else: # numerico
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1))
    
    arquivo_transicoes = f'transicoes_{nome_analise}.csv'
    n_estados = len(estados)
    
    # Definir as colunas para o DataFrame final
    colunas_df = ['video_id'] + [f'{origem}-{destino}' for origem in estados for destino in estados]

    # .rglob busca recursivamente pelo arquivo de transições
    for transicoes_csv_path in base_path.rglob(arquivo_transicoes):
        video_data_path = transicoes_csv_path.parent.parent # Pasta do vídeo
        
        try:
            # Carregar o arquivo de contagem de transições
            transicoes_matriz = pd.read_csv(transicoes_csv_path)
            
            # Gerar o vetor VMG (ex: 9 elementos para 3x3) para este vídeo
            vmg_video_flat = gerar_vmg_flatten_video(transicoes_matriz, estados)
            lista_vmg_videos.append(vmg_video_flat)

            # Carregar o 'video_id' do arquivo de informações do vídeo
            video_df = pd.read_csv(f"{video_data_path}/videos_info.csv")
            video_id = video_df['video_id'].iloc[0]
            
            # Preparar a linha de dados para o DataFrame
            dados_linha = {'video_id': video_id}

            # Preencher os valores da matriz achatada (ex: 'POS-POS', 'POS-NEU', ...)
            for i in range(vmg_video_flat.size):
                nome_coluna = colunas_df[i+1] # Pula a coluna 'video_id'
                dados_linha[nome_coluna] = vmg_video_flat[i]
            lista_info_videos.append(dados_linha)

        except FileNotFoundError:
            console.print(f"[yellow]Aviso: Arquivo 'videos_info.csv' não encontrado em {video_data_path}. Pulando vídeo.[/yellow]")
        except Exception as e:
            console.print(f"[red]Erro ao processar {transicoes_csv_path}: {e}[/red]")

    # Consolidar e salvar dados
    if not lista_vmg_videos:
        console.print(f"[yellow]Nenhum dado de transição ('{arquivo_transicoes}') encontrado para {youtuber}.[/yellow]")
        return None, None, None

    # Converter as listas em numpy array e DataFrame
    matriz_vmg_youtuber = np.array(lista_vmg_videos)
    df_youtuber = pd.DataFrame(lista_info_videos, columns=colunas_df)
    
    # Salvar o DataFrame no caminho de saída
    df_youtuber.to_csv(output_csv_path, index=False)
    console.print(f"Dados de VMG salvos em: [green]{output_csv_path}[/green]")
    
    return matriz_vmg_youtuber, df_youtuber, output_csv_path

'''
    Função para encontrar os melhores parâmetros (eps, min_samples) para o DBSCAN
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @return tuple - A melhor configuração (eps, min_samples) encontrada
'''
def otimizar_DBSCAN(matriz_vmg: np.ndarray) -> tuple:
    console.print("Buscando melhores configurações (DBSCAN)...")

    # É uma boa prática escalar os dados antes de algoritmos baseados em distância
    scaler = StandardScaler()
    X = scaler.fit_transform(matriz_vmg)

    # Parâmetros a testar
    eps_values = np.linspace(0.001, 10.0, 100)
    min_samples_values = range(2, 40)

    melhor_silhouette = -1
    melhor_config = None
    melhor_labels = None

    # Busca em grade (Grid Search)
    for eps, min_samples in itertools.product(eps_values, min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Silhouette score requer pelo menos 2 clusters (ignora ruído -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue

        try:
            sil = silhouette_score(X, labels)
            if sil > melhor_silhouette:
                melhor_silhouette = sil
                melhor_config = (eps, min_samples)
                melhor_labels = labels
        except ValueError:
            # Ocorre se todos os pontos forem ruído
            continue

    if melhor_config:
        console.print(f"✅ Melhor configuração encontrada:")
        console.print(f"   eps = {melhor_config[0]:.2f}, min_samples = {melhor_config[1]}")
        console.print(f"   Silhouette Score = {melhor_silhouette:.4f}")
        console.print(f"   Número de clusters = {len(set(melhor_labels)) - (1 if -1 in melhor_labels else 0)}\n")
    else:
        console.print("[yellow]Não foi possível encontrar uma configuração válida de clusters.[/yellow]")

    return melhor_config

'''
    Função para aplicar o DBSCAN aos dados, salvar os resultados e plotar
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @param df_vmg - O DataFrame correspondente (para salvar os labels)
    @param output_csv_path - O caminho do arquivo CSV para salvar os resultados
    @param eps - Parâmetro 'eps' do DBSCAN
    @param min_samples - Parâmetro 'min_samples' do DBSCAN
    @param analise_alvo - Nome da análise alvo para o título do gráfico
    @return np.ndarray - Os rótulos (labels) dos clusters
'''
def aplicar_DBSCAN(matriz_vmg: np.ndarray, df_vmg: pd.DataFrame, output_csv_path: Path, eps: float, min_samples: int, analise_alvo: str) -> np.ndarray:        
    # Escalar os dados para o DBSCAN e para o PCA
    scaler = StandardScaler()
    matriz_scaled = scaler.fit_transform(matriz_vmg)
    
    # Criar e ajustar o modelo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(matriz_scaled)

    # Mostrar resultados no console
    console.print("--- Resultados do DBSCAN ---")
    console.print(f"Rótulos dos clusters: {labels}")
    console.print(f"Número de clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)}")
    console.print(f"Número de ruídos (label = -1): {np.sum(labels == -1)}")

    # Atualizar o DataFrame com os rótulos dos clusters
    df_vmg['Cluster_DBSCAN'] = labels

    # Salvar o DataFrame atualizado no caminho correto
    df_vmg.to_csv(output_csv_path, index=False)
    console.print(f"Resultados dos clusters salvos em: [green]{output_csv_path}[/green]")

    # Visualizar os clusters em 2D com PCA
    # pca = PCA(n_components=2)
    # X_2d = pca.fit_transform(matriz_scaled)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='plasma', s=60)
    # plt.title(f"Clusters DBSCAN para (redução PCA) - {output_csv_path.parent.parent.name}")
    # plt.xlabel("Componente Principal 1")
    # plt.ylabel("Componente Principal 2")

    X_2d = PCA(n_components=2).fit_transform(matriz_scaled)

    cmap = plt.get_cmap('tab10')  # até 10 cores bem separadas
    unique_labels = np.unique(labels)

    plt.figure(figsize=(16, 9))
    for cluster_id in unique_labels:
        if cluster_id == -1:
            # Ruído em cinza
            cor = 'lightgray'
            label = 'Ruído'
        else:
            cor = cmap(cluster_id % 10)  # repete as 10 cores se tiver mais
            label = f'Cluster {cluster_id}'
        plt.scatter(
            X_2d[labels == cluster_id, 0],
            X_2d[labels == cluster_id, 1],
            color=cor, s=60, label=label
        )

    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Clusters DBSCAN para {analise_alvo} (redução PCA) - {output_csv_path.parent.parent.name}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Salvar o gráfico na mesma pasta de agrupamento
    plot_path = output_csv_path.parent / (output_csv_path.stem + '_dbscan_plot.png')
    plt.savefig(plot_path, dpi=500)
    console.print(f"Gráfico de clusters salvo em: [green]{plot_path}[/green]")

    return labels

'''
    Função principal para orquestrar o pipeline de agrupamento para todos os youtubers
    @param youtubers_list - Lista de youtubers a serem analisados
    @param analise_alvo - Tipo de análise (Ex: sentimento, toxicidade)
'''
def pipeline_dbcsan(youtubers_list: list[str], analise_alvo: str):
    try:
        config_metrica = METRICAS_CONFIG[analise_alvo]
    except KeyError:
        console.print(f"[bold red]Erro: A análise '{analise_alvo}' não está definida em METRICAS_CONFIG.[/bold red]")
        return

    # Iterar sobre cada youtuber da lista
    for youtuber in youtubers_list:
        console.print(f"\n--- [bold magenta]Processando Pipeline de Agrupamento com DBSCAN para: {youtuber}[/bold magenta] ---")
        
        # Preparar os dados (Gerar VMGs de todos os vídeos)
        matriz_vmg, df_vmg, csv_path = preparar_dados_agrupamento(
            youtuber, 
            analise_alvo, 
            config_metrica
        )
        
        # Pular para o próximo youtuber se nenhum dado for encontrado
        if matriz_vmg is None:
            console.print(f"[yellow]Nenhum dado de VMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
            continue
        
        # Otimizar Parâmetros do DBSCAN
        melhor_config = otimizar_DBSCAN(matriz_vmg)
        
        # Pular se a otimização não encontrar clusters válidos
        if melhor_config is None:
            console.print(f"[yellow]Não foi possível encontrar uma configuração de cluster válida para {youtuber}.[/yellow]")
            continue
            
        eps, min_samples = melhor_config
        
        # Aplicar Clusterização DBSCAN
        aplicar_DBSCAN(matriz_vmg, df_vmg, csv_path, eps=eps, min_samples=min_samples, analise_alvo=analise_alvo)

    console.print("\n--- [bold green]Pipeline de Agrupamento Concluído para todos os YouTubers[/bold green] ---")

'''
    Função para encontrar o número 'k' ótimo de clusters para o K-Means
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @param max_k - O número máximo de clusters 'k' a ser testado
    @param youtuber_name - Nome do youtuber (para títulos de gráficos/arquivos)
    @param analise_alvo - Nome da análise alvo (para títulos de gráficos/arquivos)
    @param output_dir - Pasta onde o gráfico de diagnóstico será salvo
    @return int - O número 'k' ótimo encontrado (baseado no maior score de silhueta)
'''
def otimizar_KMeans(matriz_vmg: np.ndarray, max_k: int, youtuber_name: str, analise_alvo: str, output_dir: Path) -> int:
    console.print("Buscando melhor 'k' para K-Means (Cotovelo & Silhueta)...")
    
    # Escalar os dados é crucial para K-Means e PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(matriz_vmg)
    
    # Garantir que max_k não seja maior que o número de amostras
    if max_k >= X.shape[0]:
        console.print(f"[yellow]Aviso: max_k ({max_k}) é maior ou igual ao nº de vídeos ({X.shape[0]}). Ajustando max_k para {X.shape[0] - 1}.[/yellow]")
        max_k = X.shape[0] - 1
    
    if max_k < 2:
        console.print("[yellow]Não há vídeos suficientes para formar 2 clusters. Pulando otimização.[/yellow]")
        return None

    k_range = range(2, max_k + 1)
    inercia = []
    silhouette_scores = []

    # Testar todos os valores de k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(X)
        inercia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Gerar os gráficos de diagnóstico
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Diagnóstico K-Means para os Vídeos de "{youtuber_name}" ({analise_alvo})', fontsize=18)

    # Gráfico do Método do Cotovelo
    axes[0].plot(k_range, inercia, 'bo-')
    axes[0].set_xlabel('Número de Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inércia (WCSS)', fontsize=12)
    axes[0].set_title('Método do Cotovelo (Elbow Method)', fontsize=14)
    axes[0].grid(True)

    # Gráfico do Score de Silhueta
    axes[1].plot(k_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('Número de Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Score Médio de Silhueta', fontsize=12)
    axes[1].set_title('Análise de Silhueta (Silhouette Index)', fontsize=14)
    axes[1].grid(True)
    
    # Salvar o gráfico de diagnóstico
    plot_path = output_dir / f'kmeans_diagnostico_plot_{analise_alvo}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    console.print(f"Gráfico de diagnóstico K-Means salvo em: [green]{plot_path}[/green]")
    
    # Retornar o k que deu o maior score de silhueta
    k_otimo = k_range[np.argmax(silhouette_scores)]
    console.print(f"✅ Melhor 'k' encontrado (via Silhueta): {k_otimo} (Score: {max(silhouette_scores):.4f})\n")
    return k_otimo

'''
    Função para aplicar o K-Means aos dados, salvar os resultados e plotar
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @param df_vmg - O DataFrame correspondente (para salvar os labels)
    @param output_csv_path - O caminho do arquivo CSV para salvar os resultados
    @param k_otimo - O número de clusters a ser usado
    @param analise_alvo - Nome da análise alvo para o título do gráfico
    @return np.ndarray - Os rótulos (labels) dos clusters
'''
def aplicar_KMeans(matriz_vmg: np.ndarray, df_vmg: pd.DataFrame, output_csv_path: Path, k_otimo: int, analise_alvo: str) -> np.ndarray:
    # Escalar os dados
    scaler = StandardScaler()
    matriz_scaled = scaler.fit_transform(matriz_vmg)
    
    # Criar e ajustar o modelo K-Means com o 'k' ótimo
    kmeans = KMeans(n_clusters=k_otimo, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(matriz_scaled)

    # Mostrar resultados no console
    console.print("--- Resultados do K-Means ---")
    console.print(f"Rótulos dos clusters: {labels}")
    console.print(f"Número de clusters aplicados: {k_otimo}")

    # Atualizar o DataFrame com os rótulos dos clusters
    df_vmg['Cluster_KMeans'] = labels
    
    # Salvar o DataFrame atualizado (sobrescrevendo o arquivo base de cluster)
    df_vmg.to_csv(output_csv_path, index=False)
    console.print(f"Resultados dos clusters K-Means salvos em: [green]{output_csv_path}[/green]")

    # Visualizar os clusters em 2D com PCA
    X_2d = PCA(n_components=2).fit_transform(matriz_scaled)
    
    # Contar vídeos por cluster para a legenda
    contagem_clusters = pd.Series(labels).value_counts()
    
    plt.figure(figsize=(16, 9))
    cmap = plt.get_cmap('tab10')
    unique_labels = np.unique(labels)

    # Plotar cada cluster separadamente
    for cluster_id in unique_labels:
        cor = cmap(cluster_id % 10)
        # Contagem para a legenda
        count = contagem_clusters[cluster_id]
        label = f'Cluster {cluster_id} ({count} vídeos)'
        
        plt.scatter(
            X_2d[labels == cluster_id, 0],
            X_2d[labels == cluster_id, 1],
            color=cor, s=60, label=label, alpha=0.8, edgecolors='black', linewidths=0.5
        )

    plt.legend(title="Clusters (Nº de Vídeos)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Clusters K-Means (k={k_otimo}) para {analise_alvo} (redução PCA) - {output_csv_path.parent.parent.name}")
    plt.xlabel("Componente 1"); plt.ylabel("Componente 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Salvar o gráfico na mesma pasta de agrupamento
    plot_path = output_csv_path.parent / (output_csv_path.stem + f'_kmeans_plot_k{k_otimo}.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    console.print(f"Gráfico de clusters K-Means salvo em: [green]{plot_path}[/green]")

    return labels

'''
    Função principal para orquestrar o pipeline de agrupamento K-Means
    @param youtubers_list - Lista de youtubers a serem analisados
    @param analise_alvo - Tipo de análise (Ex: sentimento, negatividade_3)
    @param max_k - Número máximo de clusters 'k' a ser testado
'''
def pipeline_kmeans(youtubers_list: list[str], analise_alvo: str, max_k: int = 10):
    try:
        config_metrica = METRICAS_CONFIG[analise_alvo]
    except KeyError:
        console.print(f"[bold red]Erro: A análise '{analise_alvo}' não está definida em METRICAS_CONFIG.[/bold red]")
        return

    # Iterar sobre cada youtuber da lista
    for youtuber in youtubers_list:
        console.print(f"\n--- [bold magenta]Processando Pipeline de Agrupamento com K-MEANS para: {youtuber}[/bold magenta] ---")
        
        # Preparar os dados (Gerar VMGs de todos os vídeos)
        matriz_vmg, df_vmg, csv_path = preparar_dados_agrupamento(
            youtuber, 
            analise_alvo, 
            config_metrica
        )
        
        if matriz_vmg is None:
            console.print(f"[yellow]Nenhum dado de VMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
            continue
        
        # Otimizar Parâmetros do K-Means (encontrar 'k' ótimo)
        k_otimo = otimizar_KMeans(matriz_vmg, max_k, youtuber, analise_alvo, csv_path.parent)
        
        if k_otimo is None:
            console.print(f"[yellow]Não foi possível encontrar um 'k' ótimo para {youtuber}.[/yellow]")
            continue
            
        # Aplicar Clusterização K-Means
        aplicar_KMeans(matriz_vmg, df_vmg, csv_path, k_otimo=k_otimo, analise_alvo=analise_alvo)

    console.print(f"\n--- [bold green]Pipeline de Agrupamento K-MEANS Concluído para '{analise_alvo}'[/bold green] ---")

if __name__ == "__main__":
    youtubers_list = ['Julia MineGirl']
    
    #analise_alvo = 'sentimento' 
    analise_alvo = 'toxicidade' 

    #pipeline_dbcsan(youtubers_list, analise_alvo)
    pipeline_kmeans(youtubers_list, analise_alvo, max_k=4)