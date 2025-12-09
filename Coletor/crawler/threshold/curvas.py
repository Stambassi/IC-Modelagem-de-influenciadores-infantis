import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

console = Console()

# Configuração
BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_TOXICIDADE = 'toxicity'

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

    # Calcular a probabilidade acumulada (eixo Y da CDF)
    n = len(sorted_data)
    y_cdf = np.arange(1, n + 1) / n
    
    # CCDF é simplesmente 1 - CDF
    y_ccdf = 1 - y_cdf
    
    return sorted_data, y_cdf, y_ccdf

'''
    Função para plotar as curvas CDF e CCDF e salvar o gráfico
    @param dados_dict - Dicionário de dados da curva de distribuição
    @param nome_grupo - Nome do grupo a ser plotado
    @param log_x - Define se o eixo x terá escala logarítmica
    @param log_y_ccdf - Define se o eixo y terá escala logarítmica para a CCDF
'''
def plotar_curvas(dados_dict: dict, nome_grupo: str, log_x: bool = False, log_y_ccdf: bool = True):
    if not dados_dict:
        console.print(f"[red]Nenhum dado para plotar para {nome_grupo}.[/red]")
        return

    # Criar figura com 2 subplots (CDF e CCDF)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Análise de Distribuição Acumulada - {nome_grupo}', fontsize=16)

    # Cores para diferenciar se houver múltiplos youtubers
    for label, dados in dados_dict.items():
        if len(dados) == 0: 
            continue
        
        x, y_cdf, y_ccdf = calcular_curvas(dados)
        
        # Plot 1: CDF
        ax1.plot(x, y_cdf, marker='.', linestyle='none', markersize=2, label=label)
        
        # Plot 2: CCDF
        ax2.plot(x, y_ccdf, marker='.', linestyle='none', markersize=2, label=label)

    # Configuração do eixo X
    for ax in [ax1, ax2]:
        ax.set_xlabel('Score de Toxicidade (x)', fontsize=12)
        ax.grid(True, which="both", ls="-", alpha=0.5)
        if log_x:
            ax.set_xscale('log')
        else:
            ax.set_xlim(0, 1.05)

    # Configuração específica do CDF
    ax1.set_title('CDF (Distribuição Acumulada)\n$P(X \leq x)$', fontsize=14)
    ax1.set_ylabel('Probabilidade', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    
    # Linhas de referência na CDF (90%, 99%)
    ax1.axhline(0.90, color='orange', linestyle='--', alpha=0.7, label='90%')
    ax1.axhline(0.99, color='red', linestyle='--', alpha=0.7, label='99%')
    ax1.legend()

    # Configuração específica CCDF
    ax2.set_title('CCDF (Cauda da Distribuição)\n$P(X > x)$', fontsize=14)
    ax2.set_ylabel('Probabilidade Complementar', fontsize=12)
    
    if log_y_ccdf:
        ax2.set_yscale('log')
        ax2.set_ylabel('Probabilidade Complementar (Log)', fontsize=12)

        # Limite Y para focar na cauda (ex: de 10^-4 até 1)
        ax2.set_ylim(bottom=1/len(x)) # Limite inferior baseado no tamanho da amostra

    # Linhas de referência na CCDF (10%, 1%, 0.1%)
    ax2.axhline(0.10, color='orange', linestyle='--', alpha=0.7, label='10% (0.1)')
    ax2.axhline(0.01, color='red', linestyle='--', alpha=0.7, label='1% (0.01)')
    ax2.legend()

    plt.tight_layout()
    
    # Salvar
    output_dir = BASE_FOLDER / 'threshold' / 'curvas'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"curvas_distribuicao_{nome_grupo.lower().replace(' ', '_')}.png"
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"Gráfico salvo: [green]{output_path}[/green]")

if __name__ == "__main__":
    lista_youtubers = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer']
    
    # Plotar a curva agregada de todos os youtubers
    dados_geral = carregar_dados(lista_youtubers, agregado=True)
    plotar_curvas(dados_geral, "Geral Agregado", log_y_ccdf=True)
    
    # Plotar individual (um gráfico para cada youtuber)
    dados_individuais = carregar_dados(lista_youtubers, agregado=False)
    for yt, dados in dados_individuais.items():
        plotar_curvas({yt: dados}, yt, log_y_ccdf=True)
        
    # Plotar comparativo (todos no mesmo gráfico)
    plotar_curvas(dados_individuais, "Comparativo_Youtubers", log_y_ccdf=True)