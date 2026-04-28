from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console
import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

METRICAS_CONFIG = {
    'pysentimiento': {
        'coluna_base': 'sentimento_dominante', 
        'tipo_estados': 'categorico', 
        'estados': ['POS', 'NEU', 'NEG']
    },
    'negatividade': {
        'coluna_base': 'negatividade',
        'tipo_estados': 'numerico',
        'n_estados': 3 
    },
    'detoxify': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', 
        'limiares': [0.0, 0.20, 0.80, 1.01], 
        'estados': ['NT', 'GZ', 'T']
    },
    'perspective': {
        'coluna_base': 'p_toxicity',
        'tipo_estados': 'numerico_categorizado', 
        'limiares': [0.0, 0.20, 0.40, 1.01], 
        'estados': ['NT', 'GZ', 'T']
    }
}

'''
    Função para calcular e persistir as Matrizes de Transição (VMG) para cada subgrupo (cluster) identificado por um algoritmo
    
    @param escopo - O alvo do agrupamento (ex: 'Geral', 'Minecraft', 'Julia MineGirl')
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'detoxify', 'pysentimiento')
    @param metrica_agrupamento - A métrica que originou o agrupamento (ex: 'confianca')
    @param nome_algoritmo - O algoritmo utilizado (ex: 'KMeans', 'DBSCAN')
    @param metricas_calculo - Lista de métricas a serem calculadas para o cluster (ex: ['lift', 'confianca', 'coeficiente_variacao'])
'''
def calcular_matrizes_dos_clusters(
    escopo: str, 
    mapa_categorias: dict, 
    metrica_config: dict, 
    nome_analise: str, 
    metrica_agrupamento: str, 
    nome_algoritmo: str, 
    metricas_calculo: list[str]
) -> None:
    console.print(f"\n[bold blue]>>> Calculando matrizes internas dos clusters - Escopo: {escopo} ({nome_algoritmo})[/bold blue]")

    # 1. Definir os caminhos base dependendo do escopo
    if escopo == 'Geral' or escopo in mapa_categorias.values():
        base_dir_agrupamento = Path(f'files/VMG/{escopo}/Agrupamento')
        youtubers_alvo = list(mapa_categorias.keys()) if escopo == 'Geral' else [yt for yt, cat in mapa_categorias.items() if cat == escopo]
    else:
        base_dir_agrupamento = Path(f'files/{escopo}/VMG/Agrupamento')
        youtubers_alvo = [escopo]

    # Arquivo de agrupamento gerado anteriormente
    csv_cluster_path = base_dir_agrupamento / f'cluster_{nome_analise}_{metrica_agrupamento}_{nome_algoritmo.lower()}.csv'
    
    if not csv_cluster_path.exists():
        console.print(f"[yellow]Aviso: Arquivo de clusters não encontrado em {csv_cluster_path}. Pulando.[/yellow]")
        return

    # 2. Carregar os labels dos clusters
    df_clusters = pd.read_csv(csv_cluster_path)
    coluna_label = f'Cluster_{nome_algoritmo}'
    if coluna_label not in df_clusters.columns:
        console.print(f"[red]Erro: Coluna {coluna_label} não encontrada no CSV.[/red]")
        return

    # 3. Carregar todas as transições brutas do escopo
    lista_transicoes = []
    for youtuber in youtubers_alvo:
        base_path_yt = Path(f'files/{youtuber}')
        if not base_path_yt.is_dir(): 
            continue

        for p in base_path_yt.rglob(f'VMG/Matrizes/transicoes_{nome_analise}.csv'):
            try:
                df_t = pd.read_csv(p)
                if not df_t.empty:
                    df_t['youtuber'] = youtuber
                    df_t['video_id'] = p.parent.parent.parent.name
                    lista_transicoes.append(df_t)
            except Exception:
                continue

    if not lista_transicoes:
        return

    df_todas_transicoes = pd.concat(lista_transicoes, ignore_index=True)

    # Definir os tipos categóricos
    if metrica_config['tipo_estados'] in ['categorico', 'numerico_categorizado']:
        estados = metrica_config['estados']
    else:
        estados = list(range(1, metrica_config['n_estados'] + 1))

    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
    df_todas_transicoes['estado'] = df_todas_transicoes['estado'].astype(tipo_categorico)
    df_todas_transicoes['proximo_estado'] = df_todas_transicoes['proximo_estado'].astype(tipo_categorico)

    # 4. Separar as métricas solicitadas
    metricas_absolutas = [m for m in metricas_calculo if m in ['probabilidade', 'suporte', 'confianca', 'lift']]
    metricas_estatisticas = [m for m in metricas_calculo if m in ['media', 'desvio_padrao', 'coeficiente_variacao']]
    funcs_estatisticas = {
        'media': 'mean',
        'desvio_padrao': 'std',
        'coeficiente_variacao': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }

    # 5. Iterar sobre cada cluster único e calcular as matrizes
    clusters_unicos = df_clusters[coluna_label].unique()

    for cluster_id in clusters_unicos:
        # Nome da pasta do cluster
        nome_pasta_cluster = 'Outliers' if cluster_id == -1 else f'Cluster_{cluster_id}'
        
        # Filtra quais vídeos pertencem a este cluster
        videos_do_cluster = df_clusters[df_clusters[coluna_label] == cluster_id]
        
        # Faz um merge/filtro para pegar apenas as transições dos vídeos deste cluster
        # Usa 'youtuber' e 'video_id' como chave composta para evitar conflitos de nomes iguais em canais diferentes
        df_escopo = df_todas_transicoes.merge(videos_do_cluster[['youtuber', 'video_id']], on=['youtuber', 'video_id'], how='inner')
        
        if df_escopo.empty: continue

        # Diretório de saída
        dir_saida_matrizes = base_dir_agrupamento / 'Clusters' / f'{nome_algoritmo}_{nome_analise}_{metrica_agrupamento}' / nome_pasta_cluster / 'Matrizes'
        dir_saida_matrizes.mkdir(parents=True, exist_ok=True)

        try:
            # --- CÁLCULO DAS MÉTRICAS ABSOLUTAS ---
            if metricas_absolutas:
                df_counts = df_escopo.groupby(['estado', 'proximo_estado'], observed=False)['contagem'].sum().reset_index()
                n_total = df_counts['contagem'].sum()
                n_u = df_counts.groupby('estado', observed=False)['contagem'].transform('sum')
                n_v = df_counts.groupby('proximo_estado', observed=False)['contagem'].transform('sum')

                resultados = {}
                if 'suporte' in metricas_absolutas: resultados['suporte'] = (df_counts['contagem'] / n_total).fillna(0)
                if 'confianca' in metricas_absolutas: resultados['confianca'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'probabilidade' in metricas_absolutas: resultados['probabilidade'] = (df_counts['contagem'] / n_u).fillna(0)
                if 'lift' in metricas_absolutas:
                    confianca_base = (df_counts['contagem'] / n_u).fillna(0)
                    p_v = n_v / n_total
                    resultados['lift'] = (confianca_base / p_v).fillna(0)

                for m_name, values in resultados.items():
                    df_counts[m_name] = values
                    matriz = df_counts.pivot(index='estado', columns='proximo_estado', values=m_name).fillna(0)
                    matriz.to_csv(dir_saida_matrizes / f'VMG_{nome_analise}_{m_name}.csv')

            # --- CÁLCULO DAS MÉTRICAS ESTATÍSTICAS ---
            if metricas_estatisticas:
                # Cria um ID único combinando youtuber e video
                df_escopo['video_id_unique'] = df_escopo['youtuber'] + '_' + df_escopo['video_id']
                n_u_video = df_escopo.groupby(['video_id_unique', 'estado'], observed=False)['contagem'].transform('sum')
                df_escopo['prob_video'] = (df_escopo['contagem'] / n_u_video).fillna(0)

                for m_name in metricas_estatisticas:
                    func = funcs_estatisticas[m_name]
                    df_stat = df_escopo.groupby(['estado', 'proximo_estado'], observed=False)['prob_video'].agg(func).reset_index()
                    matriz = df_stat.pivot(index='estado', columns='proximo_estado', values='prob_video').fillna(0)
                    matriz.to_csv(dir_saida_matrizes / f'VMG_{nome_analise}_{m_name}.csv')

            console.print(f"   [green]✓ Matrizes do {nome_pasta_cluster} geradas com sucesso![/green]")

        except Exception as e:
            console.print(f"[red]Erro ao processar {nome_pasta_cluster} em {escopo}: {e}[/red]")

'''
    Função para gerar um Heatmap que mostra as métricas de transição de forma adaptativa.

    @param matrix_path - Caminho para o arquivo CSV da matriz VTMG
    @param output_path - Caminho onde a imagem será salva
    @param title - Título do gráfico
    @param metrica - Nome da métrica sendo plotada
'''
def gerar_heatmap_vmg(matrix_path: Path, output_path: Path, title: str, metrica: str):
    try:
        df_matrix = pd.read_csv(matrix_path, index_col=0)
        plt.figure(figsize=(10, 8))
        
        # Ajuste dinâmico de escala e cores dependendo da natureza da métrica
        if metrica in ['probabilidade', 'confianca', 'suporte', 'media']:
            v_min, v_max = 0.0, 1.0
            cmap = "Blues"
        elif metrica == 'lift':
            v_min, v_max = 0.0, None
            cmap = "YlOrRd"
        elif metrica == 'desvio_padrao':
            v_min, v_max = 0.0, None
            cmap = "Purples" # Roxo para dispersão absoluta
        elif metrica == 'coeficiente_variacao':
            v_min, v_max = 0.0, None
            cmap = "Oranges" # Laranja para variação relativa
        else:
            v_min, v_max = None, None
            cmap = "Greys"
        
        ax = sns.heatmap(
            df_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap, 
            linewidths=.5,
            vmin=v_min, vmax=v_max,
            cbar_kws={'label': metrica.replace('_', ' ').title()}
        )
        
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel("Estado de Destino (V)", fontsize=12)
        ax.set_ylabel("Estado de Origem (U)", fontsize=12)
        
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        console.print(f"[red]Erro no heatmap ({metrica}): {e}[/red]")

'''
    Função para gerar Heatmaps das Matrizes de Transição (VMG) geradas internamente em cada cluster
    
    @param escopo - O alvo do agrupamento (ex: 'Geral', 'Minecraft', 'Julia MineGirl')
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param nome_analise - O nome da análise (ex: 'detoxify', 'pysentimiento')
    @param metrica_agrupamento - A métrica que originou o agrupamento (ex: 'confianca')
    @param nome_algoritmo - O algoritmo utilizado (ex: 'KMeans', 'DBSCAN')
    @param metricas_calculo - Lista de métricas a serem plotadas
'''
def gerar_plots_dos_clusters(
    escopo: str, 
    mapa_categorias: dict, 
    nome_analise: str, 
    metrica_agrupamento: str, 
    nome_algoritmo: str, 
    metricas_calculo: list[str]
) -> None:    
    # 1. Definir o caminho base do Agrupamento do escopo
    if escopo == 'Geral' or escopo in mapa_categorias.values():
        base_dir_clusters = Path(f'files/VMG/{escopo}/Agrupamento/Clusters/{nome_algoritmo}_{nome_analise}_{metrica_agrupamento}')
    else:
        base_dir_clusters = Path(f'files/{escopo}/VMG/Agrupamento/Clusters/{nome_algoritmo}_{nome_analise}_{metrica_agrupamento}')

    if not base_dir_clusters.is_dir():
        return

    # 2. Percorrer cada pasta de Cluster existente
    for pasta_cluster in base_dir_clusters.iterdir():
        if not pasta_cluster.is_dir(): 
            continue   
            

        dir_matrizes = pasta_cluster / 'Matrizes'
        dir_plots = pasta_cluster / 'Plots'
        dir_plots.mkdir(exist_ok=True)

        for metrica in metricas_calculo:
            matriz_path = dir_matrizes / f'VMG_{nome_analise}_{metrica}.csv'
            
            if matriz_path.exists():
                heatmap_path = dir_plots / f'heatmap_{nome_analise}_{metrica}.png'
                
                # Ex: "VTMG Cluster_0 (Minecraft) - Lift | Base: Kmeans (Confianca)"
                titulo = f"VTMG {pasta_cluster.name} ({escopo}) - {metrica.replace('_', ' ').title()}\nBase: {nome_algoritmo} ({metrica_agrupamento.title()})"
                
                try:
                    # Reutiliza a função base 'gerar_heatmap_vmg' 
                    gerar_heatmap_vmg(matriz_path, heatmap_path, titulo, metrica)
                except Exception as e:
                    console.print(f"   [red]Erro ao gerar heatmap para {pasta_cluster.name} ({metrica}): {e}[/red]")

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
    nome_analise = 'detoxify'
    # nome_analise = 'perspective'
    metrica_base = 'probabilidade'
    # nome_algoritmo = 'KMeans'
    nome_algoritmo = 'DBSCAN'

    # Setup do escopo
    # escopos = ["Geral", "Minecraft", "Roblox"]
    # escopos = list(mapa_youtubers_categoria.keys())
    escopos = ["Geral", "Minecraft", "Roblox"] + list(mapa_youtubers_categoria.keys())

    # Garante que o iterador seja sempre uma lista
    if not isinstance(escopos, list):
        escopos = [escopos]

    metricas_desejadas = ['probabilidade', 'media', 'desvio_padrao', 'coeficiente_variacao']

    for escopo in escopos:
        console.print(f"[bold magenta]===== Processando Pipeline de Plotagem para: {escopo} =====[/bold magenta]")

        calcular_matrizes_dos_clusters(
            escopo=escopo,
            mapa_categorias=mapa_youtubers_categoria,
            metrica_config=METRICAS_CONFIG['detoxify'],
            nome_analise=nome_analise,
            metrica_agrupamento=metrica_base,
            nome_algoritmo=nome_algoritmo,
            metricas_calculo=metricas_desejadas
        )

        gerar_plots_dos_clusters(
            escopo=escopo,
            mapa_categorias=mapa_youtubers_categoria,
            nome_analise=nome_analise,
            metrica_agrupamento=metrica_base,
            nome_algoritmo=nome_algoritmo,
            metricas_calculo=metricas_desejadas
        )

        console.print(f"\n[bold magenta]===== Pipeline de Agrupamento Concluído para o escopo {escopo} =====[/bold magenta]\n")