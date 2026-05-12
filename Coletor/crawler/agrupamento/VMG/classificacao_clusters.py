import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

# Dicionário de configuração mantido para consistência do pipeline
METRICAS_CONFIG = {
    'detoxify': {'estados': ['NT', 'GZ', 'T']},
    'perspective': {'estados': ['NT', 'GZ', 'T']},
    'pysentimiento': {'estados': ['POS', 'NEU', 'NEG']}
}

'''
    Função para validar e imprimir a proporção de vídeos "Non-Toxic" (sem transição para o estado tóxico alvo) em cada cluster

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param algoritmo_cluster - O algoritmo utilizado no agrupamento (ex: 'KMeans', 'DBSCAN')
    @param estado_alvo - O estado considerado como tóxico ou indesejado (padrão: 'T')
'''
def relatorio_pureza_clusters(df_vmg: pd.DataFrame, algoritmo_cluster: str, estado_alvo: str = 'T') -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    
    # Filtra as colunas que representam chegadas ao estado alvo (ex: termina com '->T')
    colunas_destino_alvo = [col for col in df_vmg.columns if col.endswith(f'->{estado_alvo}')]
    
    if not colunas_destino_alvo:
        console.print(f"[red]Erro: Nenhuma coluna de destino para o estado '{estado_alvo}' encontrada.[/red]")
        return

    # Um vídeo é considerado "puro/non-toxic" se a soma das transições para o estado T for zero
    quantidade_toxico = df_vmg[colunas_destino_alvo].sum(axis=1)
    df_vmg['is_non_toxic'] = quantidade_toxico == 0

    clusters = sorted(df_vmg[cluster_col].unique())
    total_non_toxic_geral = df_vmg['is_non_toxic'].sum()

    # Criação da Tabela para o Console
    tabela = Table(title=f"Análise de Pureza (Vídeos sem transição para '{estado_alvo}')", show_header=True, header_style="bold magenta")
    tabela.add_column("Cluster", justify="center")
    tabela.add_column("Tamanho do Cluster", justify="right")
    tabela.add_column("Qtd. Non-Toxic", justify="right")
    tabela.add_column("Pureza do Cluster (%)", justify="right", style="green")
    tabela.add_column("Representação Global (%)", justify="right", style="cyan")

    for cluster_id in clusters:
        cluster_df = df_vmg[df_vmg[cluster_col] == cluster_id]
        cluster_size = len(cluster_df)
        
        non_toxic_count = cluster_df['is_non_toxic'].sum()
        
        perc_pureza = (non_toxic_count / cluster_size * 100) if cluster_size > 0 else 0
        perc_global = (non_toxic_count / total_non_toxic_geral * 100) if total_non_toxic_geral > 0 else 0

        tabela.add_row(
            str(cluster_id),
            str(cluster_size),
            str(non_toxic_count),
            f"{perc_pureza:.2f}%",
            f"{perc_global:.2f}%"
        )

    console.print(tabela)

'''
    Função para gerar e persistir o histograma de distribuição de transições/visitas em cada estado por cluster

    @param df_vmg - DataFrame contendo as matrizes VMG achatadas e os rótulos dos clusters
    @param algoritmo_cluster - O algoritmo utilizado no agrupamento (ex: 'KMeans', 'DBSCAN')
    @param analise - O nome da análise para exibição no título (ex: 'detoxify', 'perspective')
    @param estados - Lista de estados da métrica configurada (ex: ['NT', 'GZ', 'T'])
    @param output_path - Caminho (Path) onde a imagem do gráfico será salva
'''
def gerar_grafico_distribuicao_visitas(df_vmg: pd.DataFrame, algoritmo_cluster: str, analise: str, estados: list[str], output_path: Path) -> None:
    cluster_col = f'Cluster_{algoritmo_cluster}'
    clusters = sorted(df_vmg[cluster_col].unique())

    metrics = []
    # Cria as somatórias dinamicamente com base nos estados configurados
    for estado in estados:
        colunas_destino = [col for col in df_vmg.columns if col.endswith(f'->{estado}')]
        nome_coluna_soma = f'quantidade_visitas_{estado}'
        df_vmg[nome_coluna_soma] = df_vmg[colunas_destino].sum(axis=1)
        metrics.append((f"Chegadas em '{estado}'", nome_coluna_soma))

    n_clusters = len(clusters)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(nrows=n_clusters, ncols=n_metrics, figsize=(6 * n_metrics, 4 * n_clusters))
    
    # Garante que 'axes' seja sempre uma matriz 2D para indexação segura, mesmo se houver apenas 1 cluster
    axes = np.atleast_2d(axes)

    for i, cluster_id in enumerate(clusters):
        subset = df_vmg[df_vmg[cluster_col] == cluster_id]

        for j, (label, col) in enumerate(metrics):
            ax = axes[i, j]
            data = subset[col]

            # Define os bins como números inteiros, lidando com o fato de que a máxima pode ser zero
            max_val = int(data.max()) if pd.notna(data.max()) else 0
            bins = range(max_val + 2)
            
            ax.hist(data, bins=bins, edgecolor='black', color=sns.color_palette("muted")[j])

            if i == 0: ax.set_title(label, fontweight='bold')
            if j == 0: ax.set_ylabel(f'Cluster {cluster_id}\n(Nº de Vídeos)', fontweight='bold')
            
            ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle(f'Distribuição de Transições de Chegada por Cluster\n({algoritmo_cluster} | {analise.capitalize()})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Salva na pasta de agrupamento em vez de apenas mostrar na tela
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    console.print(f"   [green]✓ Gráfico de distribuição salvo em: {output_path}[/green]")

'''
    Função principal para orquestrar a leitura dos dados, análise de pureza e geração dos gráficos de distribuição dos clusters

    @param escopo - O alvo do agrupamento a ser analisado (ex: 'Geral', 'Minecraft', 'Roblox')
    @param mapa_categorias - Dicionário mapeando {nome_youtuber: categoria}
    @param nome_analise - O nome da análise a ser lida (ex: 'detoxify', 'perspective')
    @param metrica_agrupamento - A métrica que originou o agrupamento atual (ex: 'contagem', 'probabilidade')
    @param algoritmo - O algoritmo de clusterização utilizado (ex: 'KMeans', 'DBSCAN')
'''
def analisar_pureza_distribuicao(
    escopo: str, 
    mapa_categorias: dict, 
    nome_analise: str, 
    metrica_agrupamento: str, 
    algoritmo: str
) -> None:
    console.print(f"\n[bold blue]>>> Analisando Pureza de Clusters - Escopo: {escopo} ({nome_analise} | {metrica_agrupamento})[/bold blue]")

    # 1. Resolução Dinâmica de Diretórios (mantendo a arquitetura do pipeline)
    if escopo == 'Geral' or escopo in mapa_categorias.values():
        base_dir_agrupamento = Path(f'files/VMG/{escopo}/Agrupamento')
    else:
        base_dir_agrupamento = Path(f'files/{escopo}/VMG/Agrupamento')

    csv_cluster_path = base_dir_agrupamento / f'cluster_{nome_analise}_{metrica_agrupamento}_{algoritmo.lower()}.csv'
    
    if not csv_cluster_path.exists():
        console.print(f"[yellow]Aviso: Arquivo {csv_cluster_path.name} não encontrado. Execute o agrupamento primeiro.[/yellow]")
        return

    # 2. Leitura dos Dados
    df_vmg = pd.read_csv(csv_cluster_path)
    
    # Descobre o estado alvo (geralmente o último da lista, ex: 'T')
    estados_config = METRICAS_CONFIG.get(nome_analise, {}).get('estados', ['NT', 'GZ', 'T'])
    estado_toxico_alvo = estados_config[-1]

    # 3. Execução das Análises
    relatorio_pureza_clusters(df_vmg, algoritmo, estado_alvo=estado_toxico_alvo)
    
    # 4. Geração dos Gráficos
    plot_output_path = base_dir_agrupamento / 'Plots' / f'hist_distribuicao_{nome_analise}_{metrica_agrupamento}_{algoritmo.lower()}.png'
    gerar_grafico_distribuicao_visitas(df_vmg, algoritmo, nome_analise, estados_config, plot_output_path)

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
    escopos = ['Geral', 'Minecraft', 'Roblox']
    analises = ['perspective', 'detoxify']
    algoritmos = ['KMeans']
    
    metrica_base = 'contagem' 

    for escopo in escopos:
        for analise in analises:
            for alg in algoritmos:
                analisar_pureza_distribuicao(
                    escopo=escopo,
                    mapa_categorias=mapa_youtubers_categoria,
                    nome_analise=analise,
                    metrica_agrupamento=metrica_base,
                    algoritmo=alg
                )