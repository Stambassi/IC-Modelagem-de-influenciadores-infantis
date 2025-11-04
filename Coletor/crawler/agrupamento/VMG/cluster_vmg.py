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
from rich.console import Console

console = Console()

'''
    Função para gerar a matriz de transição (VSMG) para um único vídeo e achatá-la para um vetor
    @param df_transicoes - DataFrame com as contagens de transição
    @param n_estados - O número de estados (ex: 3 para uma matriz 3x3)
    @return numpy.ndarray - O VSMG como um vetor unidimensional (N*N elementos)
'''
def gerar_vsmg_flatten_video(df_transicoes: pd.DataFrame, n_estados: int) -> np.ndarray:
    # Calcular as probabilidades de transição
    somas_por_estado = df_transicoes.groupby('estado')['contagem'].transform('sum')
    df_transicoes = df_transicoes.copy()
    df_transicoes['probabilidade'] = (df_transicoes['contagem'] / somas_por_estado).fillna(0)

    # Criar os rótulos de estado (ex: [1, 2, 3])
    labels = list(range(1, n_estados + 1))
    
    # Pivotar os dados para o formato de matriz
    matriz = df_transicoes.pivot(index='estado', columns='proximo_estado', values='probabilidade')
    
    # Reindexar a matriz para garantir que ela seja N x N.
    # Se uma transição (ex: 3 -> 1) nunca ocorreu, ela será preenchida com 0.0.
    matriz = matriz.reindex(index=labels, columns=labels, fill_value=0.0)
    
    # Achatar (flatten) a matriz para um vetor 1D
    # Converte a matriz (ex: 3x3) em um vetor (ex: 9 elementos)
    vetor_vsmg = matriz.to_numpy().flatten()
    
    return vetor_vsmg

'''
    Função para processar todos os vídeos de um youtuber, gerar o VSMG de cada um e salvar os dados
    @param youtuber - Nome do youtuber a ser processado
    @param metrica - A métrica base da análise (ex: 'negatividade')
    @param n_estados - O número de estados (ex: 3)
    @return tuple - Contendo a matriz de VSMGs (vídeos x features), o DataFrame e o caminho do arquivo salvo
'''
def preparar_dados_agrupamento(youtuber: str, metrica: str, n_estados: int) -> tuple: 
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
    output_csv_path = output_dir / f'vsmg_cluster_data_{metrica}_{n_estados}.csv'

    # Coletar VSMG de cada vídeo
    lista_vsmg_videos = [] # Lista para os dados numpy (para o clustering)
    lista_info_videos = [] # Lista para os dados do DataFrame (para o CSV)
    
    # Definir o nome do arquivo de transições a ser procurado
    arquivo_transicoes = f'transicoes_{metrica}_{n_estados}.csv'
    
    # Definir as colunas para o DataFrame final
    colunas_df = ['video_id'] + [f'{i}-{j}' for i in range(1, n_estados + 1) for j in range(1, n_estados + 1)]

    # .rglob busca recursivamente pelo arquivo de transições
    for transicoes_csv_path in base_path.rglob(arquivo_transicoes):
        video_data_path = transicoes_csv_path.parent.parent # Pasta do vídeo
        
        try:
            # Carregar o arquivo de contagem de transições
            transicoes_matriz = pd.read_csv(transicoes_csv_path)
            
            # Gerar o vetor VSMG (ex: 9 elementos) para este vídeo
            vsmg_video_flat = gerar_vsmg_flatten_video(transicoes_matriz, n_estados)
            lista_vsmg_videos.append(vsmg_video_flat)

            # Carregar o 'video_id' do arquivo de informações do vídeo
            video_df = pd.read_csv(f"{video_data_path}/videos_info.csv")
            video_id = video_df['video_id'].iloc[0]
            
            # Preparar a linha de dados para o DataFrame
            dados_linha = {'video_id': video_id}

            # Preencher os valores da matriz achatada (ex: '1-1', '1-2', ...)
            for i in range(vsmg_video_flat.size):
                row = i // n_estados + 1
                col = i % n_estados + 1
                dados_linha[f'{row}-{col}'] = vsmg_video_flat[i]
            lista_info_videos.append(dados_linha)

        except FileNotFoundError:
            console.print(f"[yellow]Aviso: Arquivo 'videos_info.csv' não encontrado em {video_data_path}. Pulando vídeo.[/yellow]")
        except Exception as e:
            console.print(f"[red]Erro ao processar {transicoes_csv_path}: {e}[/red]")

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
    @param matriz_vsmg - A matriz de dados (vídeos x features)
    @return tuple - A melhor configuração (eps, min_samples) encontrada
'''
def otimizar_DBSCAN(matriz_vsmg: np.ndarray) -> tuple:
    console.print("Buscando melhores configurações (DBSCAN)...")

    # É uma boa prática escalar os dados antes de algoritmos baseados em distância
    scaler = StandardScaler()
    X = scaler.fit_transform(matriz_vsmg)

    # Parâmetros a testar
    eps_values = np.linspace(0.1, 2.0, 20)
    min_samples_values = range(2, 10)

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
    @param matriz_vsmg - A matriz de dados (vídeos x features)
    @param df_vsmg - O DataFrame correspondente (para salvar os labels)
    @param output_csv_path - O caminho do arquivo CSV para salvar os resultados
    @param eps - Parâmetro 'eps' do DBSCAN
    @param min_samples - Parâmetro 'min_samples' do DBSCAN
    @return np.ndarray - Os rótulos (labels) dos clusters
'''
def aplicar_DBSCAN(matriz_vsmg: np.ndarray, df_vsmg: pd.DataFrame, output_csv_path: Path, eps: float, min_samples: int) -> np.ndarray:    
    # Escalar os dados para o DBSCAN e para o PCA
    scaler = StandardScaler()
    matriz_scaled = scaler.fit_transform(matriz_vsmg)
    
    # Criar e ajustar o modelo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(matriz_scaled)

    # Mostrar resultados no console
    console.print("--- Resultados do DBSCAN ---")
    console.print(f"Rótulos dos clusters: {labels}")
    console.print(f"Número de clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)}")
    console.print(f"Número de ruídos (label = -1): {np.sum(labels == -1)}")

    # Atualizar o DataFrame com os rótulos dos clusters
    df_vsmg['Cluster_DBSCAN'] = labels
    # Salvar o DataFrame atualizado no caminho correto
    df_vsmg.to_csv(output_csv_path, index=False)
    console.print(f"Resultados dos clusters salvos em: [green]{output_csv_path}[/green]")

    # Visualizar os clusters em 2D com PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(matriz_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='plasma', s=60)
    plt.title(f"Clusters DBSCAN (redução PCA) - {output_csv_path.parent.parent.name}")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    
    # Salvar o gráfico na mesma pasta de agrupamento
    plot_path = output_csv_path.parent / (output_csv_path.stem + '_dbscan_plot.png')
    plt.savefig(plot_path)
    console.print(f"Gráfico de clusters salvo em: [green]{plot_path}[/green]")

    return labels

'''
    Função principal para orquestrar o pipeline de agrupamento para todos os youtubers.
'''
def main():
    # Configurações da Análise
    YOUTUBERS_LIST = ['Tex HS', 'Julia MineGirl']
    METRICA = 'negatividade'
    N_ESTADOS = 3

    # Iterar sobre cada youtuber da lista
    for youtuber in YOUTUBERS_LIST:
        console.print(f"\n--- [bold magenta]Processando Pipeline de Agrupamento para: {youtuber}[/bold magenta] ---")
        
        # Preparar os dados (Gerar VSMGs de todos os vídeos)
        matriz_vsmg, df_vsmg, csv_path = preparar_dados_agrupamento(youtuber, METRICA, N_ESTADOS)
        
        # Pular para o próximo youtuber se nenhum dado for encontrado
        if matriz_vsmg is None:
            console.print(f"[yellow]Nenhum dado de VSMG encontrado para {youtuber}. Pulando agrupamento.[/yellow]")
            continue
        
        # Otimizar Parâmetros do DBSCAN
        melhor_config = otimizar_DBSCAN(matriz_vsmg)
        
        # Pular se a otimização não encontrar clusters válidos
        if melhor_config is None:
            console.print(f"[yellow]Não foi possível encontrar uma configuração de cluster válida para {youtuber}.[/yellow]")
            continue
            
        eps, min_samples = melhor_config
        
        # Aplicar Clusterização DBSCAN
        aplicar_DBSCAN(matriz_vsmg, df_vsmg, csv_path, eps=eps, min_samples=min_samples)

    console.print("\n--- [bold green]Pipeline de Agrupamento Concluído para todos os YouTubers[/bold green] ---")

if __name__ == "__main__":
    main()