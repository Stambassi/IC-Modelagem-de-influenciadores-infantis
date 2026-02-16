import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from rich.console import Console

console = Console()

# Configuração
BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_TOXICIDADE = 'toxicity'

# Mapeamento de categorias para estilos de linha e análise
MAPA_YOUTUBERS_CATEGORIA = {
    'Amy Scarlet': 'Roblox',
    'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft',
    'Julia MineGirl': 'Roblox',
    'Kass e KR': 'Minecraft',
    'Lokis': 'Roblox',
    'Luluca Games': 'Roblox',
    'Papile': 'Roblox',
    'Robin Hood Gamer': 'Minecraft',
    'TazerCraft': 'Minecraft',
    # 'Tex HS': 'Misto'
}

'''
    Função para carregar os dados de toxicidade de uma lista de youtubers
    @param youtubers_lista - Lista de youtubers a serem analisados
    @return np.ndarray - Lista de listas de toxicidade de cada vídeo
'''
def carregar_dados(youtubers_list: list, agregado: bool = True):
    console.print("[bold]Carregando dados de toxicidade...[/bold]")
    dados_dict = {}
    todos_scores = []
    
    for youtuber in youtubers_list:
        path = BASE_FOLDER / youtuber
        if not path.is_dir(): continue
        
        scores_youtuber = []
        for file in path.rglob(INPUT_FILENAME):
            try:
                df = pd.read_csv(file)
                if COLUNA_TOXICIDADE in df.columns:
                    # Filtra apenas dados válidos
                    vals = df[COLUNA_TOXICIDADE].dropna().tolist()
                    scores_youtuber.extend(vals)
            except:
                pass
        
        if scores_youtuber:
            dados_dict[youtuber] = np.array(scores_youtuber)
            todos_scores.extend(scores_youtuber)
            
    if agregado:
        return {"Geral": np.array(todos_scores)}
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
    @param log_x - Define se o eixo x terá escala logarítmica
    @param log_y_ccdf - Define se o eixo y terá escala logarítmica para a CCDF
'''
def plotar_curvas(dados_dict: dict, nome_grupo: str, log_y_ccdf: bool = True):
    if not dados_dict:
        console.print(f"[red]Nenhum dado para plotar para {nome_grupo}.[/red]")
        return
    
    # Agrupamento por Categoria
    if nome_grupo == "Categorias":
        dados_agrupados = {}
        for yt, scores in dados_dict.items():
            cat = MAPA_YOUTUBERS_CATEGORIA.get(yt, 'Desconhecido')
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
    fig.suptitle(f'Distribuição de Toxicidade: {nome_grupo}', fontsize=16, fontweight='bold')

    # Dicionário de estilos por categoria para facilitar a comparação visual
    estilos_categoria = {
        'Minecraft': '-',     # Linha sólida
        'Roblox': '--',       # Linha tracejada
        'Misto': ':',         # Linha pontilhada
        'Desconhecido': '-.'  # Traço-ponto
    }

    for i, (label, dados) in enumerate(dados_dict.items()):
        if len(dados) == 0: continue
        
        x, y_cdf, y_ccdf = calcular_curvas(dados)
        color = colors[i % len(colors)]
        
        # Identifica a categoria para definir o estilo da linha
        categoria = MAPA_YOUTUBERS_CATEGORIA.get(label, 'Desconhecido')
        estilo = estilos_categoria.get(categoria, '-')
        
        # Plot 1: CDF - Uso de steps-post para representar a natureza acumulada real
        if nome_grupo == "Categorias":
            ax1.plot(x, y_cdf, label=f"{label}", color=color, linestyle=estilo, linewidth=1.5, drawstyle='steps-post')
        else:
            ax1.plot(x, y_cdf, label=f"{label} ({categoria})", color=color, linestyle=estilo, linewidth=1.5, drawstyle='steps-post')
        
        # Plot 2: CCDF - Na CCDF, filtra zeros para não quebrar o log
        mask = y_ccdf > 0
        ax2.plot(x[mask], y_ccdf[mask], color=color, linestyle=estilo, linewidth=1.5, drawstyle='steps-post')

    # Configurações dos eixos X
    for ax in [ax1, ax2]:
        ax.set_xlabel('Score de Toxicidade (Detoxify)', fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.grid(True, which="both", ls="-", alpha=0.3)

    # Configuração específica do CDF
    ax1.set_title(f'CDF - $P(X \leq x)$', fontsize=13)
    ax1.set_ylabel('Probabilidade Acumulada', fontsize=11)
    ax1.set_ylim(0, 1.02)
    
    # Linha de referência na CDF (90%) - Adicionada anotação direta para limpar a legenda
    ax1.axhline(0.90, color='gray', linestyle=':', alpha=0.6)
    ax1.text(0.02, 0.91, '90%', color='gray', fontsize=9, fontweight='bold')
    
    # Configuração específica CCDF
    ax2.set_title('CCDF - $P(X > x)$', fontsize=13)
    if log_y_ccdf:
        ax2.set_yscale('log')
        # Formatação de potência de 10 (ex: 10^-1, 10^-2)
        ax2.set_ylabel('Probabilidade (Escala Log)', fontsize=11)

    # Legenda inteligente: Se forem muitos, coloca fora do gráfico para não obstruir os dados
    if len(dados_dict) > 1:
        ax1.legend(loc='upper left', bbox_to_anchor=(0, -0.18), ncol=3, fontsize=9, frameon=True)
    else:
        ax1.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Salvar
    output_dir = BASE_FOLDER / 'threshold' / 'curvas'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"curvas_distribuicao_{nome_grupo.lower().replace(' ', '_')}.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"Gráfico salvo com diferenciação por categoria: [green]{output_path}[/green]")

if __name__ == "__main__":
    # Lista de youtubers baseada no mapa de categorias
    lista_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    # Plotar a curva agregada de todos os youtubers
    dados_geral = carregar_dados(lista_youtubers, agregado=True)
    plotar_curvas(dados_geral, "Geral Agregado", log_y_ccdf=True)
    
    # Plotar individual (um gráfico para cada youtuber)
    dados_individuais = carregar_dados(lista_youtubers, agregado=False)
    for yt, dados in dados_individuais.items():
        plotar_curvas({yt: dados}, yt, log_y_ccdf=True)
        
    # Plotar comparativo (todos no mesmo gráfico com diferenciação visual por jogo)
    plotar_curvas(dados_individuais, "Comparativo Youtubers", log_y_ccdf=True)

    # Plotar a diferença de categorias
    plotar_curvas(dados_individuais, "Categorias", log_y_ccdf=True)