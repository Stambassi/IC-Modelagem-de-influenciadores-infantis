import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

console = Console()

# Configuração
BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'

# Mapeamento de categorias para estilos de linha e análise
MAPA_YOUTUBERS_CATEGORIA = {
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
    'TazerCraft': 'Minecraft'
}

# Configuração dos modelos de toxicidade
MODELOS_CONFIG = {
    'detoxify': {
        'coluna': 'toxicity',
        'subpasta': 'detoxify',
        'titulo': 'Detoxify'
    },
    'perspective': {
        'coluna': 'p_toxicity',
        'subpasta': 'perspective',
        'titulo': 'Perspective API'
    }
}

'''
    Função para carregar os dados de toxicidade de uma lista de youtubers
    @param youtubers_list - Lista de youtubers a serem analisados
    @param coluna - Nome da coluna de toxicidade a ser lida do CSV
    @param agregado - Se True, junta todos os scores em uma única categoria
    @return dict - Dicionário contendo as listas de toxicidade
'''
def carregar_dados(youtubers_list: list, coluna: str, agregado: bool = True):
    console.print(f"[bold]Loading toxicity data ({coluna})...[/bold]")
    dados_dict = {}
    todos_scores = []
    
    for youtuber in youtubers_list:
        path = BASE_FOLDER / youtuber
        if not path.is_dir(): continue
        
        scores_youtuber = []
        for file in path.rglob(INPUT_FILENAME):
            try:
                df = pd.read_csv(file)
                if coluna in df.columns:
                    # Filtra apenas dados válidos
                    vals = df[coluna].dropna().tolist()
                    scores_youtuber.extend(vals)
            except:
                pass
        
        if scores_youtuber:
            dados_dict[youtuber] = np.array(scores_youtuber)
            todos_scores.extend(scores_youtuber)
            
    if agregado:
        return {"General": np.array(todos_scores)}
    else:
        return dados_dict

'''
    Função para calcular os valores de X e Y para a CDF e CCDF
'''
def calcular_curvas(dados: np.ndarray):
    # Ordenar os dados
    sorted_data = np.sort(dados)
    n = len(sorted_data)
    
    # ECDF: P(X <= x)
    y_cdf = np.arange(1, n + 1) / n
    
    # CCDF: P(X > x) calculado de trás para frente para evitar erros de ponto flutuante
    y_ccdf = np.arange(n - 1, -1, -1) / n
    
    return sorted_data, y_cdf, y_ccdf

'''
    Função para plotar as curvas CDF e CCDF e salvar o gráfico
    @param dados_dict - Dicionário de dados da curva de distribuição
    @param nome_grupo - Nome do grupo a ser plotado
    @param modelo_key - Chave do modelo de análise ('detoxify' ou 'perspective')
    @param log_y_ccdf - Define se o eixo y terá escala logarítmica para a CCDF
'''
def plotar_curvas(dados_dict: dict, nome_grupo: str, modelo_key: str, log_y_ccdf: bool = True):
    if not dados_dict:
        console.print(f"[red]No data to plot for {nome_grupo} with {modelo_key}.[/red]")
        return
    
    config = MODELOS_CONFIG[modelo_key]
    coluna_titulo = config['titulo']
    subpasta = config['subpasta']
    
    # Tradução de nomes de grupo para títulos em inglês no artigo
    nomes_grupos_en = {
        "Geral Agregado": "Overall Aggregated",
        "Comparativo Youtubers": "YouTuber Comparison",
        "Categorias": "Content Categories"
    }
    nome_grupo_en = nomes_grupos_en.get(nome_grupo, nome_grupo)
    
    # Agrupamento por Categoria
    if nome_grupo == "Categorias":
        dados_agrupados = {}
        for yt, scores in dados_dict.items():
            cat = MAPA_YOUTUBERS_CATEGORIA.get(yt, 'Unknown')
            if cat not in dados_agrupados:
                dados_agrupados[cat] = []
            dados_agrupados[cat].extend(scores)
        
        # Converte listas de volta para np.array para compatibilidade com calcular_curvas
        dados_dict = {cat: np.array(s) for cat, s in dados_agrupados.items()}

    # Uso de um estilo mais limpo e paleta de cores estendida
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.tab20.colors 

    # Criar figura com 2 subplots (CDF e CCDF)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    fig.suptitle(f'Toxicity Score Distribution ({coluna_titulo}): {nome_grupo_en}', fontsize=16, fontweight='bold')

    # Dicionário de estilos por categoria para facilitar a comparação visual
    estilos_categoria = {
        'Minecraft': '-',     # Linha sólida
        'Roblox': '--',       # Linha tracejada
        'Mixed': ':',         # Linha pontilhada
        'Unknown': '-.'       # Traço-ponto
    }

    for i, (label, dados) in enumerate(dados_dict.items()):
        if len(dados) == 0: continue
        
        x, y_cdf, y_ccdf = calcular_curvas(dados)
        color = colors[i % len(colors)]
        
        # Identifica a categoria para definir o estilo da linha e a legenda correta
        if nome_grupo == "Categorias":
            categoria = label
            estilo = estilos_categoria.get(categoria, '-')
            legenda_label = label
        else:
            categoria = MAPA_YOUTUBERS_CATEGORIA.get(label, 'Unknown')
            estilo = estilos_categoria.get(categoria, '-')
            legenda_label = f"{label} ({categoria})"
        
        # Plot 1: CDF - Uso de steps-post para representar a natureza acumulada real (aumentado linewidth para 2.0 para o revisor)
        ax1.plot(x, y_cdf, label=legenda_label, color=color, linestyle=estilo, linewidth=2.0, drawstyle='steps-post')
        
        # Plot 2: CCDF - Na CCDF, filtra zeros para não quebrar o log
        mask = y_ccdf > 0
        ax2.plot(x[mask], y_ccdf[mask], color=color, linestyle=estilo, linewidth=2.0, drawstyle='steps-post')

    # Configurações dos eixos X
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'Toxicity Score ({coluna_titulo})', fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.tick_params(labelsize=10)

    # Configuração específica do CDF
    ax1.set_title(r'CDF - $P(X \leq x)$', fontsize=14)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.set_ylim(0, 1.02)
    
    # Linha de referência na CDF (90%) - Adicionada anotação direta para limpar a legenda
    ax1.axhline(0.90, color='gray', linestyle=':', alpha=0.6)
    ax1.text(0.02, 0.91, '90%', color='gray', fontsize=10, fontweight='bold')
    
    # Configuração específica CCDF
    ax2.set_title(r'CCDF - $P(X > x)$', fontsize=14)
    if log_y_ccdf:
        ax2.set_yscale('log')
        # Formatação de potência de 10
        ax2.set_ylabel('Probability (Log Scale)', fontsize=12)

    # Legenda inteligente: Se forem muitos, coloca fora do gráfico para não obstruir os dados
    if len(dados_dict) > 1:
        ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.18), ncol=3, fontsize=10, frameon=True)
    else:
        ax1.legend(fontsize=11)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Salvar
    output_dir = BASE_FOLDER / 'threshold' / subpasta
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"distribution_curves_{nome_grupo_en.lower().replace(' ', '_')}.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"[{coluna_titulo}] Saved plot to: [green]{output_path}[/green]")

if __name__ == "__main__":
    # Lista de youtubers baseada no mapa de categorias
    lista_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    for modelo_key in MODELOS_CONFIG.keys():
        console.print(f"\n[bold cyan]=== PROCESSING CURVES FOR: {modelo_key.upper()} ===[/bold cyan]")
        coluna_nome = MODELOS_CONFIG[modelo_key]['coluna']
        
        # Plotar a curva agregada de todos os youtubers
        dados_geral = carregar_dados(lista_youtubers, coluna_nome, agregado=True)
        plotar_curvas(dados_geral, "Geral Agregado", modelo_key, log_y_ccdf=True)
        
        # Plotar individual (um gráfico para cada youtuber)
        dados_individuais = carregar_dados(lista_youtubers, coluna_nome, agregado=False)
        for yt, dados in dados_individuais.items():
            plotar_curvas({yt: dados}, yt, modelo_key, log_y_ccdf=True)
            
        # Plotar comparativo (todos no mesmo gráfico com diferenciação visual por jogo)
        plotar_curvas(dados_individuais, "Comparativo Youtubers", modelo_key, log_y_ccdf=True)

        # Plotar a diferença de categorias
        plotar_curvas(dados_individuais, "Categorias", modelo_key, log_y_ccdf=True)