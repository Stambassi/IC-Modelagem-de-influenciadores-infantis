import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype
import numpy as np

import itertools
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

import re

console = Console()

# Define as regras para cada tipo de análise (métrica)
METRICAS_CONFIG = {
    'sentimento': {
        'coluna_base': 'sentimento_dominante', # Coluna no 'tiras_video.csv' que define o estado
        'tipo_estados': 'categorico', # 'categorico' (ex: POS, NEU) ou 'numerico' (ex: 0.0-1.0)
        'estados': ['POS', 'NEU', 'NEG'] # Lista de estados para análises categóricas
    },
    'negatividade': {
        'coluna_base': 'sentimento_dominante', # Coluna no 'tiras_video.csv' que define o estado
        'tipo_estados': 'categorico', # 'categorico' (ex: POS, NEU) ou 'numerico' (ex: 0.0-1.0)
        'estados': ['POS', 'NEU', 'NEG'] # Lista de estados para análises categóricas
    },
    # 'negatividade': {
    #     'coluna_base': 'negatividade',
    #     'tipo_estados': 'numerico',
    #     'n_estados': 3 # Número de 'bins' para dividir o score (ex: 3 estados)
    # },
    'toxicidade': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', # Novo tipo de estado
        'limiares': [0.0, 0.30, 0.70, 1.01], # 1.01 para garantir que 1.0 seja incluído
        'estados': ['NT', 'GZ', 'T'] # Nomes dos estados
    }
}


def iso_duration_to_seconds(duration: str) -> int:
    """
    Convert ISO 8601 duration (e.g., 'PT1H2M3S', 'PT18M24S', 'PT45S') into total seconds.
    Supports hours, minutes, and seconds.
    """
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.fullmatch(duration)
    
    if not match:
        raise ValueError(f"Invalid duration format: {duration}")
    
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


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
    # print(df_transicoes)
    matriz = df_transicoes.pivot(index='estado', columns='proximo_estado', values='probabilidade')
    
    # Reindexar a matriz para garantir que ela seja N x N
    matriz = matriz.reindex(index=labels, columns=labels, fill_value=0.0)
    
    # Achatar (flatten) a matriz para um vetor 1D
    vetor_vmg = matriz.to_numpy().flatten()
    
    return vetor_vmg


def filtrar_colunas(df, colunas_indesejadas):
    """
    Remove as colunas listadas em 'colunas_indesejadas' do DataFrame.
    """
    df_filtrado = df.drop(columns=[c for c in colunas_indesejadas if c in df.columns])
    return df_filtrado

'''
    Função para processar todos os vídeos de um youtuber, gerar o VSMG de cada um e salvar os dados
    @param youtuber - Nome do youtuber a ser processado
    @param metrica - A métrica base da análise (ex: 'negatividade')
    @param config_metrica - Métricas (ex: 3)
    @return tuple - Contendo a matriz de VSMGs (vídeos x features), o DataFrame e o caminho do arquivo salvo
'''
def preparar_dados_agrupamento(youtuber: str, metrica: str, config_metrica) -> tuple: 
    n_estados = len(config_metrica['estados'])
    console.print(f'>>> Processando VSMGs para [cyan]{youtuber}[/cyan] (Métrica: {metrica}, n={n_estados})')
    
    # Criar um objeto Path para o diretório base do youtuber
    base_path = Path(f'files/{youtuber}')

    # Se o diretório do youtuber não existir, pular
    if not base_path.is_dir():
        console.print(f"[red]Diretório não encontrado para {youtuber}. Pulando.[/red]")
        return None, None, None

    # Salvar os arquivos na pasta 'agrupamento' de cada youtuber
    output_dir = base_path / 'agrupamento'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f'vsmg_cluster_data_{metrica}.csv'

    # Coletar VSMG de cada vídeo
    lista_vsmg_videos = [] # Lista para os dados numpy (para o clustering)
    lista_info_videos = [] # Lista para os dados do DataFrame (para o CSV)
    
    # Definir o nome do arquivo de transições a ser procurado
    arquivo_transicoes = f'transicoes_{metrica}.csv'
    
    # Definir as colunas para o DataFrame final
    colunas_df = (
          ['video_id']
        + ['duration']
        + ['view_count']
        + [f'{i}-{j}' for i in range(1, n_estados + 1) for j in range(1, n_estados + 1)]
        + [f'tempo_vertice_{i}' for i in range(1, n_estados + 1)]
    )

    # .rglob busca recursivamente pelo arquivo de transições
    for transicoes_csv_path in base_path.rglob(arquivo_transicoes):
        video_data_path = transicoes_csv_path.parent.parent # Pasta do vídeo
        
        try:
            # Carregar o arquivo de contagem de transições
            transicoes_matriz = pd.read_csv(transicoes_csv_path)
            
            contagem_estados = transicoes_matriz['contagem'].tolist()
            
            
            # Gerar o vetor VSMG (ex: 9 elementos) para este vídeo
            vsmg_video_flat = gerar_vmg_flatten_video(transicoes_matriz, config_metrica['estados'])
            lista_vsmg_videos.append(vsmg_video_flat)

            # Carregar o 'video_id' do arquivo de informações do vídeo
            video_df = pd.read_csv(f"{video_data_path}/videos_info.csv")
            video_id = video_df['video_id'].iloc[0]
            duration = iso_duration_to_seconds(video_df['duration'].iloc[0])
            view_count = video_df['view_count'].iloc[0]
            
            # Preparar a linha de dados para o DataFrame
            dados_linha = {'video_id': video_id, 'duration':duration, 'view_count': view_count}

            for i in range(1, n_estados + 1):
                dados_linha[f'tempo_vertice_{i}'] = 0

            for i in range(0, len(contagem_estados), n_estados):
                grupo = contagem_estados[i:i + n_estados]
                idx_vertice = i // n_estados + 1  # começa em 1
                dados_linha[f'tempo_vertice_{idx_vertice}'] = sum(grupo)

            # Preencher os valores da matriz achatada (ex: '1-1', '1-2', ...)
            for i in range(vsmg_video_flat.size):
                row = i // n_estados + 1
                col = i % n_estados + 1
                dados_linha[f'{row}-{col}'] = vsmg_video_flat[i]

            lista_info_videos.append(dados_linha)

        except FileNotFoundError:
            console.print(f"[yellow]Aviso: Arquivo 'videos_info.csv' não encontrado em {video_data_path}. Pulando vídeo.[/yellow]")
        except Exception as e:
            console.print(f"[red]Erro ao processar {transicoes_csv_path}: {e} ({type(e)})[/red]")

    # Consolidar e salvar dados
    if not lista_vsmg_videos:
        console.print(f"[yellow]Nenhum dado de transição encontrado para {youtuber}.[/yellow]")
        return None, None, None

    # Converter as listas em numpy array e DataFrame
    matriz_vsmg_youtuber = np.array(lista_vsmg_videos)
    df_youtuber = pd.DataFrame(lista_info_videos, columns=colunas_df)
    
    # Salvar o DataFrame no caminho de saída
    df_youtuber.to_csv(output_csv_path, index=False)
    console.print(f"Dados de VSMG salvos em: [green]{output_csv_path}[/green]")
    
    return matriz_vsmg_youtuber, df_youtuber, output_csv_path

'''
    Função para encontrar os melhores parâmetros (eps, min_samples) para o DBSCAN
    @param youtuber_df - A matriz de dados (vídeos x features)
    @return tuple - A melhor configuração (eps, min_samples) encontrada
'''
def otimizar_DBSCAN(youtuber_df: pd.DataFrame) -> tuple:
    console.print("Buscando melhores configurações (DBSCAN)...")

    # É uma boa prática escalar os dados antes de algoritmos baseados em distância
    X = pipeline_pre_processamento(youtuber_df)
    

    # Parâmetros a testar
    eps_values = np.linspace(0.001, 10.0, 50)
    min_samples_values = range(2, 21)

    melhor_silhouette = -1
    melhor_config = None
    melhor_labels = None

    print(" ")
    with Progress() as progress:
        task = progress.add_task(
            "[bold cyan]Progresso: [/]", 
            total=len(eps_values) * len(min_samples_values)
        )

        for eps, min_samples in itertools.product(eps_values, min_samples_values):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                progress.advance(task)
                continue

            try:
                sil = silhouette_score(X, labels)
                if sil > melhor_silhouette:
                    melhor_silhouette = sil
                    melhor_config = (eps, min_samples)
                    melhor_labels = labels
            except ValueError:
                # Pode ocorrer se todos os pontos forem ruído
                pass

            progress.advance(task)

    if melhor_config:
        console.print(f"✅ Melhor configuração encontrada:")
        console.print(f"   eps = {melhor_config[0]:.2f}, min_samples = {melhor_config[1]}")
        console.print(f"   Silhouette Score = {melhor_silhouette:.4f}")
        console.print(f"   Número de clusters = {len(set(melhor_labels)) - (1 if -1 in melhor_labels else 0)}\n")
    else:
        console.print("[yellow]Não foi possível encontrar uma configuração válida de clusters.[/yellow]")

    return melhor_config


def pipeline_pre_processamento(dados):
    scaler = RobustScaler()
    dados_sem_outlier = scaler.fit_transform(dados)
    scaler_min_max = MinMaxScaler()
    dado_normalizado = scaler_min_max.fit_transform(dados_sem_outlier)
    return dado_normalizado 


'''
    Função para aplicar o DBSCAN aos dados, salvar os resultados e plotar
    @param df_vmg - O DataFrame correspondente (para salvar os labels)
    @param output_csv_path - O caminho do arquivo CSV para salvar os resultados
    @param eps - Parâmetro 'eps' do DBSCAN
    @param min_samples - Parâmetro 'min_samples' do DBSCAN
    @param analise_alvo - Nome da análise alvo para o título do gráfico
    @return np.ndarray - Os rótulos (labels) dos clusters
'''
def aplicar_DBSCAN(df_vmg: pd.DataFrame, output_csv_path: Path, eps: float, min_samples: int, analise_alvo: str) -> np.ndarray:        
    # Escalar os dados para o DBSCAN e para o PCA
    matriz_scaled = pipeline_pre_processamento(df_vmg)
    
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

    X_2d = PCA(n_components=2).fit_transform(matriz_scaled)

    cmap = plt.get_cmap('tab10')  # até 10 cores bem separadas
    unique_labels = np.unique(labels)
    contagem_clusters = pd.Series(labels).value_counts()

    plt.figure(figsize=(16, 9))

    for cluster_id in unique_labels:
        count = contagem_clusters[cluster_id]
        if cluster_id == -1:
            # Ruído em cinza
            cor = 'lightgray'
            label = f'Ruído ({count} vídeos)'
        else:
            cor = cmap(cluster_id % 10)  # repete as 10 cores se tiver mais
            label = f'Cluster {cluster_id} ({count} vídeos)'
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

        df_vmg_filtrado = filtrar_colunas(df_vmg, ['video_id'])

        # Otimizar Parâmetros do DBSCAN
        melhor_config = otimizar_DBSCAN(df_vmg_filtrado)
        
        # Pular se a otimização não encontrar clusters válidos
        if melhor_config is None:
            console.print(f"[yellow]Não foi possível encontrar uma configuração de cluster válida para {youtuber}.[/yellow]")
            continue
            
        eps, min_samples = melhor_config
        
        # Aplicar Clusterização DBSCAN
        aplicar_DBSCAN(df_vmg_filtrado, csv_path, eps=eps, min_samples=min_samples, analise_alvo=analise_alvo)

    console.print("\n--- [bold green]Pipeline de Agrupamento Concluído para todos os YouTubers[/bold green] ---")

def pipeline_dbcsan_geral(youtubers_list: list[str], analise_alvo: str):
    try:
        config_metrica = METRICAS_CONFIG[analise_alvo]
    except KeyError:
        console.print(f"[bold red]Erro: A análise '{analise_alvo}' não está definida em METRICAS_CONFIG.[/bold red]")
        return

    console.print(f"\n--- [bold magenta]Processando Pipeline de Agrupamento com DBSCAN de todos os youtubers[/bold magenta] ---")

    dfs = []
    for youtuber in youtubers_list:
        # Preparar os dados (Gerar VMGs de todos os vídeos)
        matriz_vmg, df_vmg, csv_path = preparar_dados_agrupamento(
            youtuber, 
            analise_alvo, 
            config_metrica
        )
        df_vmg['youtuber'] = youtuber
        dfs.append(df_vmg)
    
    df_vmg_total = pd.concat(dfs, ignore_index=True)
        
    # Pular para o próximo youtuber se nenhum dado for encontrado
    if df_vmg_total is None:
        console.print(f"[yellow]Nenhum dado de VMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
        return

    df_vmg_filtrado = filtrar_colunas(df_vmg_total, ['video_id','youtuber'])

    melhor_config = otimizar_DBSCAN(df_vmg_filtrado)
        
    if melhor_config is None:
        console.print(f"[yellow]Não foi possível encontrar uma configuração de cluster válida para {youtuber}.[/yellow]")
        return
            
    eps, min_samples = melhor_config
    csv_path = Path('agrupamento/dbscan_geral.csv')    
    aplicar_DBSCAN(df_vmg_filtrado, csv_path, eps=eps, min_samples=min_samples, analise_alvo=analise_alvo)

    console.print("\n--- [bold green]Pipeline de Agrupamento Concluído para todos os YouTubers[/bold green] ---")


'''
    Função para encontrar o número 'k' ótimo de clusters para o K-Means
    @param youtuber_df - Dados do youtuber já filtrados
    @param max_k - O número máximo de clusters 'k' a ser testado
    @param youtuber_name - Nome do youtuber (para títulos de gráficos/arquivos)
    @param analise_alvo - Nome da análise alvo (para títulos de gráficos/arquivos)
    @param output_dir - Pasta onde o gráfico de diagnóstico será salvo
    @return int - O número 'k' ótimo encontrado (baseado no maior score de silhueta)
'''
def otimizar_KMeans(youtuber_df: pd.DataFrame, max_k: int, youtuber_name: str, analise_alvo: str, output_dir: Path) -> int:
    console.print("Buscando melhor 'k' para K-Means (Cotovelo & Silhueta)...")
    
    # Escalar os dados é crucial para K-Means e PCA
    X = pipeline_pre_processamento(youtuber_df)
    
    
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
def aplicar_KMeans(df_vmg: pd.DataFrame, output_csv_path: Path, k_otimo: int, analise_alvo: str) -> np.ndarray:
    # Escalar os dados
    matriz_scaled = pipeline_pre_processamento(df_vmg)
    
    # Criar e ajustar o modelo K-Means com o 'k' ótimo
    kmeans = KMeans(n_clusters=k_otimo, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(matriz_scaled)

    # Mostrar resultados no console
    console.print("--- Resultados do K-Means ---")
    console.print(f"Rótulos dos clusters: {labels}")
    console.print(f"Número de clusters aplicados: {k_otimo}")

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

        # Filtrar as colunas indesejadas para o KMeans
        df_vmg_filtrado = filtrar_colunas(df_vmg, ['video_id'])
        
        if matriz_vmg is None:
            console.print(f"[yellow]Nenhum dado de VMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
            continue
        
        # Otimizar Parâmetros do K-Means (encontrar 'k' ótimo)
        k_otimo = otimizar_KMeans(df_vmg_filtrado, max_k, youtuber, analise_alvo, csv_path.parent)
        
        if k_otimo is None:
            console.print(f"[yellow]Não foi possível encontrar um 'k' ótimo para {youtuber}.[/yellow]")
            continue
            
        # Aplicar Clusterização K-Means
        labels = aplicar_KMeans(df_vmg_filtrado, csv_path, k_otimo=k_otimo, analise_alvo=analise_alvo)

        # Atualizar o DataFrame com os rótulos dos clusters
        df_vmg['Cluster_KMeans'] = labels
        
        # Salvar o DataFrame atualizado (sobrescrevendo o arquivo base de cluster)
        df_vmg.to_csv(csv_path, index=False)

    console.print(f"\n--- [bold green]Pipeline de Agrupamento K-MEANS Concluído para '{analise_alvo}'[/bold green] ---")

def pipeline_kmeans_geral(youtubers_list, analise_alvo, max_k = 10):
    try:
        config_metrica = METRICAS_CONFIG[analise_alvo]
    except KeyError:
        console.print(f"[bold red]Erro: A análise '{analise_alvo}' não está definida em METRICAS_CONFIG.[/bold red]")
        return

    console.print(f"\n--- [bold magenta]Processando Pipeline de Agrupamento com K-MEANS para todos youtubers[/bold magenta] ---")
        
    # Preparar os dados (Gerar VMGs de todos os vídeos)
    
    dfs = []
    for youtuber in youtubers_list:
        matriz_vmg, df_vmg, csv_path = preparar_dados_agrupamento(
            youtuber, 
            analise_alvo, 
            config_metrica
        )
        df_vmg["youtuber"] = youtuber
        dfs.append(df_vmg)

    df_vmg_total = pd.concat(dfs, ignore_index=True)
    # print(df_vmg_total)
    csv_path = Path("agrupamento/geral_kmeans_data.csv")

    # Filtrar as colunas indesejadas para o KMeans
    df_vmg_filtrado = filtrar_colunas(df_vmg_total, ['video_id','youtuber'])
    
    if matriz_vmg is None:
        console.print(f"[yellow]Nenhum dado de VMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
        return
    
    # Otimizar Parâmetros do K-Means (encontrar 'k' ótimo)
    k_otimo = otimizar_KMeans(df_vmg_filtrado, max_k, youtuber, analise_alvo, csv_path.parent)
    
    if k_otimo is None:
        console.print(f"[yellow]Não foi possível encontrar um 'k' ótimo para {youtuber}.[/yellow]")
        return
        
    # Aplicar Clusterização K-Means
    labels = aplicar_KMeans(df_vmg_filtrado, csv_path, k_otimo=k_otimo, analise_alvo=analise_alvo)

    # Atualizar o DataFrame com os rótulos dos clusters
    df_vmg_total['Cluster_KMeans'] = labels
    
    # Salvar o DataFrame atualizado (sobrescrevendo o arquivo base de cluster)
    df_vmg_total.to_csv(csv_path, index=False)

    console.print(f"\n--- [bold green]Pipeline de Agrupamento K-MEANS Concluído para '{analise_alvo}'[/bold green] ---") 

if __name__ == "__main__":
    # youtubers_list = ['Julia MineGirl']
    df = pd.read_csv('youtuberslist.csv')
    df = df[df['videosTranscritos'] != 0]
    youtubers_list = df['nome'].tolist()
    youtubers_list.remove('Robin Hood Gamer')
    # analise_alvo = 'negatividade'
    analise_alvo = 'sentimento' 
    # analise_alvo = 'toxicidade' 

    # pipeline_dbcsan(youtubers_list, analise_alvo)
    pipeline_dbcsan_geral(youtubers_list,analise_alvo)
    # pipeline_kmeans(youtubers_list, analise_alvo, max_k=10)
    pipeline_kmeans_geral(youtubers_list,analise_alvo,10)
    