import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import itertools
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

'''
    Função para coletar e achatar as matrizes VMG de nível de vídeo para um determinado escopo de análise
    
    @param escopo - O alvo do agrupamento (ex: 'Geral', 'Minecraft', 'Julia MineGirl')
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param nome_analise - O nome da análise (ex: 'detoxify', 'pysentimiento')
    @param metrica_vmg - A métrica da matriz a ser utilizada para o agrupamento (ex: 'confianca')
    @return tuple - Matriz numpy achatada, DataFrame de metadados e caminho do CSV salvo
'''
def preparar_dados_agrupamento(escopo: str, mapa_categorias: dict, nome_analise: str, metrica_vmg: str) -> tuple: 
    console.print(f'>>> Coletando vídeos para agrupamento - Escopo: [cyan]{escopo}[/cyan] (Análise: {nome_analise} | Métrica: {metrica_vmg})')

    # 1. Identifica quais Youtubers pertencem ao escopo solicitado
    youtubers_alvo = []
    if escopo == 'Geral':
        youtubers_alvo = list(mapa_categorias.keys())
        output_dir = Path('files/VMG/Geral/Agrupamento')
    elif escopo in mapa_categorias.values(): # É uma Categoria (ex: Minecraft)
        youtubers_alvo = [yt for yt, cat in mapa_categorias.items() if cat == escopo]
        output_dir = Path(f'files/VMG/{escopo}/Agrupamento')
    else: # É um Youtuber específico
        youtubers_alvo = [escopo]
        output_dir = Path(f'files/{escopo}/VMG/Agrupamento')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f'cluster_{nome_analise}_{metrica_vmg}.csv'

    lista_vmg_videos = [] 
    lista_info_videos = [] 
    colunas_df = None

    padrao_busca = f'VMG/Matrizes/VMG_{nome_analise}_{metrica_vmg}.csv'

    for youtuber in youtubers_alvo:
        base_path = Path(f'files/{youtuber}')
        if not base_path.is_dir(): 
            continue

        for matriz_csv_path in base_path.rglob(padrao_busca):
            # Garante que só vamos ler as matrizes do nível do vídeo
            # O nome da pasta do nível 3 será o Youtuber ou a Categoria nas matrizes agregadas
            nome_nivel_acima = matriz_csv_path.parent.parent.parent.name
            if nome_nivel_acima in [youtuber, 'Geral', 'Minecraft', 'Roblox']:
                continue # Pula matrizes agregadas

            try:
                df_matriz = pd.read_csv(matriz_csv_path, index_col=0)
                if df_matriz.empty: continue

                vmg_video_flat = df_matriz.to_numpy().flatten()
                lista_vmg_videos.append(vmg_video_flat)

                video_id = nome_nivel_acima # O nome da pasta do vídeo
                
                if colunas_df is None:
                    estados_origem = df_matriz.index.astype(str).tolist()
                    estados_destino = df_matriz.columns.astype(str).tolist()
                    # Adiciona a coluna youtuber para não perder o rastreio em escopos macros
                    colunas_df = ['youtuber', 'video_id'] + [f'{origem}->{destino}' for origem in estados_origem for destino in estados_destino]

                dados_linha = {'youtuber': youtuber, 'video_id': video_id}
                for i, col_name in enumerate(colunas_df[2:]):
                    dados_linha[col_name] = vmg_video_flat[i]
                    
                lista_info_videos.append(dados_linha)

            except Exception as e:
                console.print(f"[red]Erro ao processar {matriz_csv_path}: {e}[/red]")

    if not lista_vmg_videos:
        console.print(f"[yellow]Nenhum dado em nível de vídeo encontrado para o escopo '{escopo}'.[/yellow]")
        return None, None, None

    matriz_vmg = np.array(lista_vmg_videos)
    df_vmg = pd.DataFrame(lista_info_videos, columns=colunas_df)
    
    df_vmg.to_csv(output_csv_path, index=False)
    console.print(f"Base de dados agrupada ({len(lista_info_videos)} vídeos) salva em: [green]{output_csv_path}[/green]")
    
    return matriz_vmg, df_vmg, output_csv_path

'''
    Função para realizar a busca em grade (Grid Search) e encontrar os melhores parâmetros para um algoritmo de agrupamento genérico
    
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @param algoritmo_class - A classe do algoritmo do scikit-learn (ex: DBSCAN, KMeans)
    @param param_grid - Dicionário com os parâmetros a testar (ex: {'eps': [0.5, 1.0], 'min_samples': [3, 4]})
    @return dict - Um dicionário com a melhor configuração de parâmetros encontrada
'''
def otimizar_parametros_agrupamento(matriz_vmg: np.ndarray, algoritmo_class, param_grid: dict) -> dict:
    console.print(f"Buscando melhores configurações para o algoritmo {algoritmo_class.__name__}...")

    scaler = StandardScaler()
    X = scaler.fit_transform(matriz_vmg)

    # Extrai os nomes dos parâmetros e suas respectivas listas de valores a testar
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    melhor_silhouette = -1
    melhor_config = None
    melhor_labels = None

    # Cria todas as combinações possíveis de parâmetros
    for valores_combinados in itertools.product(*param_values):
        # Monta um dicionário de kwargs para a iteração atual (ex: {'eps': 0.5, 'min_samples': 3})
        kwargs = dict(zip(param_names, valores_combinados))
        
        # Instancia o modelo com os parâmetros atuais
        modelo = algoritmo_class(**kwargs)
        labels = modelo.fit_predict(X)

        # Validação de número de clusters (Silhouette precisa de 2 a N-1 clusters)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2 or n_clusters >= len(X):
            continue

        try:
            sil = silhouette_score(X, labels)
            if sil > melhor_silhouette:
                melhor_silhouette = sil
                melhor_config = kwargs
                melhor_labels = labels
        except ValueError:
            continue

    if melhor_config:
        console.print(f"✅ Melhor configuração encontrada:")
        for param, val in melhor_config.items():
            console.print(f"   {param} = {val}")
        console.print(f"   Silhouette Score = {melhor_silhouette:.4f}")
        console.print(f"   Número de clusters válidos = {len(set(melhor_labels)) - (1 if -1 in melhor_labels else 0)}\n")
    else:
        console.print("[yellow]Não foi possível encontrar uma configuração válida de clusters na grade fornecida.[/yellow]")

    return melhor_config

'''
    Função para aplicar um algoritmo de agrupamento aos dados, salvar os resultados e plotar a redução PCA
    com layout dinâmico adaptativo.
    
    @param matriz_vmg - A matriz de dados (vídeos x features)
    @param df_vmg - O DataFrame correspondente (para salvar os labels)
    @param output_csv_path - O caminho do arquivo CSV base para salvar os resultados
    @param modelo_agrupamento - O modelo do scikit-learn instanciado (ex: KMeans, DBSCAN)
    @param nome_algoritmo - String com o nome do algoritmo (para títulos)
    @param escopo - O alvo do agrupamento (para o título do gráfico)
    @return np.ndarray - Os rótulos (labels) dos clusters
'''
def aplicar_agrupamento(
    matriz_vmg: np.ndarray, 
    df_vmg: pd.DataFrame, 
    output_csv_path: Path, 
    modelo_agrupamento, 
    nome_algoritmo: str, 
    escopo: str,
    nome_analise: str
) -> np.ndarray:    
    # Escalonamento
    if not nome_analise == 'contagem': 
        scaler = StandardScaler()
        matriz_scaled = scaler.fit_transform(matriz_vmg)
    else:
        matriz_scaled = matriz_vmg

    # Treino e Predição
    labels = modelo_agrupamento.fit_predict(matriz_scaled)

    console.print(f"[bold green]===== Resultados do {nome_algoritmo} ({escopo}) =====[/bold green]")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    console.print(f"Número de clusters encontrados: {n_clusters}")
    
    if -1 in labels:
        console.print(f"Número de ruídos/outliers (label = -1): {np.sum(labels == -1)}")

    # Salva em arquivo CSV
    coluna_label = f'Cluster_{nome_algoritmo}'
    df_vmg[coluna_label] = labels
    final_csv_path = output_csv_path.parent / f"{output_csv_path.stem}_{nome_algoritmo.lower()}.csv"
    df_vmg.to_csv(final_csv_path, index=False)

    # --- Visualização em 2D com PCA ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(matriz_scaled)

    # Contagem de vídeos por cluster para a legenda
    contagem_clusters = pd.Series(labels).value_counts()

    plt.figure(figsize=(16, 9))
    cmap = plt.get_cmap('tab10')
    unique_labels = np.unique(labels)

    # Plotagem individual de cada cluster
    for cluster_id in unique_labels:
        # Tratamento especial para "ruído" se o algoritmo for DBSCAN (-1)
        if cluster_id == -1:
            cor = 'dimgray'
            count = contagem_clusters[cluster_id]
            label = f'Outliers (-1) ({count} vídeos)'
            alpha_val = 0.4
        else:
            cor = cmap(cluster_id % 10)
            count = contagem_clusters[cluster_id]
            label = f'Cluster {cluster_id} ({count} vídeos)'
            alpha_val = 0.8
            
        plt.scatter(
            X_2d[labels == cluster_id, 0],
            X_2d[labels == cluster_id, 1],
            color=cor, s=60, label=label, alpha=alpha_val, edgecolors='black', linewidths=0.5
        )

    # Formatação Estética
    plt.legend(title="Clusters (Nº de Vídeos)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Título extrai a métrica (ex: 'confianca') de dentro do nome do arquivo salvo para colocar no gráfico
    metrica_nome = output_csv_path.stem.split('_')[-1].title()
    plt.title(f"Clusters {nome_algoritmo} para {escopo} (Redução PCA)\nMétrica Agrupada: {metrica_nome} ({nome_analise})")
    
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Salvamento da imagem
    plot_path = output_csv_path.parent / f"{output_csv_path.stem}_{nome_algoritmo.lower()}_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    console.print(f"Gráfico de clusters {nome_algoritmo} salvo em: [green]{plot_path}[/green]\n")

    return labels

'''
    Função principal para orquestrar o pipeline de agrupamento para todos os youtubers.
'''
if __name__ == "__main__":
    # Mapeia cada youtuber para sua categoria principal
    mapa_youtubers_categoria = {
        'Julia MineGirl': 'Roblox',
        'Papile': 'Roblox',
        'Tex HS': 'Roblox',
        'Amy Scarlet': 'Roblox',
        'Luluca Games': 'Roblox',
        'meu nome é david': 'Roblox',
        'Lokis': 'Roblox',
        
        'Robin Hood Gamer': 'Minecraft',
        'AuthenticGames': 'Minecraft',
        'Cadres': 'Minecraft',
        'Athos': 'Minecraft',
        'JP Plays': 'Minecraft',
        'Marcelodrv': 'Minecraft',
        'Geleia': 'Minecraft',
        'Kass e KR': 'Minecraft',
    }
    
    # Setup das escolhas
    # nome_analise = 'detoxify'
    nome_analise = 'perspective'
    # metrica_base = 'probabilidade'
    metrica_base = 'contagem'

    # Setup do escopo
    # escopos = ["Geral", "Minecraft", "Roblox"]
    # escopos = list(mapa_youtubers_categoria.keys())
    escopos = ["Geral", "Minecraft", "Roblox"] + list(mapa_youtubers_categoria.keys())

    # Garante que o iterador seja sempre uma lista
    if not isinstance(escopos, list):
        escopos = [escopos]

    for escopo in escopos:
        console.print(f"[bold magenta]===== Processando Pipeline de Agrupamento para: {escopo} =====[/bold magenta]")
            
        # 1. Pega os vídeos de todos os youtubers do escopo
        matriz_vmg, df_vmg, csv_path = preparar_dados_agrupamento(escopo, mapa_youtubers_categoria, nome_analise, metrica_base)

        # Verifica se foram encontrados dados antes de aplicar os algoritmos
        if matriz_vmg is None:
            console.print(f"[yellow]Aviso: Dados insuficientes ou não encontrados para '{escopo}'. Pulando para o próximo.[/yellow]\n")
            continue

        # 2. Otimiza o k-means testando diferentes k
        grid_kmeans = {'n_clusters': [2, 3, 4, 5, 6], 'n_init': ['auto'], 'random_state': [42]}
        melhor_config_km = otimizar_parametros_agrupamento(matriz_vmg, KMeans, grid_kmeans)

        # 3. Aplica com os melhores parâmetros e plota o gráfico
        if melhor_config_km:
            modelo_km = KMeans(**melhor_config_km)
            aplicar_agrupamento(matriz_vmg, df_vmg, csv_path, modelo_km, 'KMeans', escopo, nome_analise)

        # 4. Otimiza o DBSCAN utilizando diferentes parâmetros
        grid_dbscan = {'eps': np.linspace(0.5, 3.0, 15), 'min_samples': [3, 4, 5]}
        melhores_params_db = otimizar_parametros_agrupamento(matriz_vmg, DBSCAN, grid_dbscan)

        # 5. Aplica com os melhores parâmetros e plota o gráfico
        if melhores_params_db:
            modelo_db = DBSCAN(**melhores_params_db)
            aplicar_agrupamento(matriz_vmg, df_vmg, csv_path, modelo_db, 'DBSCAN', escopo, nome_analise)

        console.print(f"[bold magenta]===== Pipeline de Agrupamento Concluído para o escopo {escopo} =====[/bold magenta]\n")