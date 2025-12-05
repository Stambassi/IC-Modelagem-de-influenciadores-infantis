import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console

console = Console()

BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_TOXICIDADE = 'toxicity'

# Limite inferior da Zona Cinza (Fixo)
LIMITE_GZ_LOW = 0.30 

# Thresholds de Toxicidade para testar (Eixo de variação)
THRESHOLDS_TESTE = [0.50, 0.60, 0.70, 0.80, 0.90]

'''
    Função para carregar as sequências puras de scores de toxicidade de todos os vídeos
    @param youtuber - Nome do youtuber a ser analisado
    @return todas_sequencias - Lista de listas (cada lista interna é um vídeo)
'''
def carregar_scores_videos(youtuber: str) -> list[list[float]]:
    youtuber_path = BASE_FOLDER / youtuber
    if not youtuber_path.is_dir():
        return []
    
    todas_sequencias = []
    
    for file in youtuber_path.rglob(INPUT_FILENAME):
        try:
            df = pd.read_csv(file)
            if COLUNA_TOXICIDADE in df.columns:
                scores = df[COLUNA_TOXICIDADE].dropna().tolist()
                if scores:
                    todas_sequencias.append(scores)
        except:
            pass
            
    return todas_sequencias

'''
    Calcula apenas a coluna "Destino = Tóxico" da matriz de transição para um dado threshold
    @param sequencia_scores - Lista de listas (cada lista interna é um vídeo)
    @param threshold_t - Limite superior de toxicidade
    @return probs - Dicionário: {'NT->T': prob, 'GZ->T': prob, 'T->T': prob}
'''
def calcular_coluna_toxica(sequencias_scores: list[list[float]], threshold_t: float) -> dict:
    # Definir bins: [0, GZ_LOW, THRESHOLD_T, 1.01]
    bins = [0.0, LIMITE_GZ_LOW, threshold_t, 1.01]
    labels = ['NT', 'GZ', 'T']
    
    transicoes_count = {
        ('NT', 'T'): 0, 
        ('GZ', 'T'): 0, 
        ('T', 'T'): 0
    }
    total_saidas = {'NT': 0, 'GZ': 0, 'T': 0}
    
    for video in sequencias_scores:
        # Discretizar o vídeo inteiro de uma vez
        estados = pd.cut(video, bins=bins, labels=labels, include_lowest=True, right=False)
        
        # Iterar para contar transições
        for i in range(len(estados) - 1):
            atual = estados[i]
            proximo = estados[i+1]
            
            # Contabiliza a saída do estado atual
            total_saidas[atual] += 1
            
            # Se o próximo for T, contabiliza a transição de risco
            if proximo == 'T':
                if (atual, 'T') in transicoes_count:
                    transicoes_count[(atual, 'T')] += 1

    # Calcular probabilidades
    probs = {}
    for origem in ['NT', 'GZ', 'T']:
        if total_saidas[origem] > 0:
            p = transicoes_count.get((origem, 'T'), 0) / total_saidas[origem]
        else:
            p = 0.0
        probs[f'{origem} -> T'] = p
        
    return probs

'''
    Função para gerar o gráfico comparativo de subplots lado a lado
    @param youtuber - Nome do youtuber a ser analisado
'''
def plotar_sensibilidade_toxicidade(youtuber: str):
    console.print(f"\n[bold cyan]Gerando Análise de Sensibilidade para: {youtuber}[/bold cyan]")
    
    # Carregar dados brutos (apenas uma vez)
    scores_brutos = carregar_scores_videos(youtuber)
    if not scores_brutos:
        console.print("[red]Nenhum dado encontrado.[/red]")
        return

    # Calcular probabilidades para cada threshold
    dados_plot = []
    
    for t in THRESHOLDS_TESTE:
        probs = calcular_coluna_toxica(scores_brutos, t)
        probs['Threshold'] = t # Adiciona o threshold para referência
        dados_plot.append(probs)
    
    # Plotagem
    n_plots = len(THRESHOLDS_TESTE)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6), sharey=True)
    
    fig.suptitle(f'Sensibilidade do Risco de Toxicidade - {youtuber}\n(Probabilidade de Transição para o Estado T)', fontsize=16, weight='bold')
    
    # Define cores para as barras (NT=Verde, GZ=Laranja, T=Vermelho)
    cores = ['#2ca02c', '#ff7f0e', '#d62728'] 
    
    max_y = 0 # Para ajustar o limite Y dinamicamente

    for i, dados in enumerate(dados_plot):
        ax = axes[i] if n_plots > 1 else axes
        threshold = dados['Threshold']
        
        # Extrair valores para plotar
        valores = [dados['NT -> T'], dados['GZ -> T'], dados['T -> T']]
        categorias = ['NT $\u2192$ T', 'GZ $\u2192$ T', 'T $\u2192$ T'] # Setas bonitas
        
        max_y = max(max_y, max(valores))
        
        # Criar barras
        bars = ax.bar(categorias, valores, color=cores, edgecolor='black', alpha=0.8)
        
        # Adicionar o valor em cima da barra
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Estilização
        ax.set_title(f'Threshold (T) = {threshold:.2f}', fontsize=12, pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:
            ax.set_ylabel('Probabilidade de Transição', fontsize=12)
    
    # Ajuste fino do eixo Y (dá uma margem de 10% acima da maior barra)
    plt.ylim(0, min(1.0, max_y * 1.15))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Espaço para o título principal
    
    # Salvar
    output_dir = BASE_FOLDER / youtuber / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'analise_sensibilidade_toxicidade.png'
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    console.print(f"[green]Gráfico salvo em: {output_path}[/green]")

if __name__ == "__main__":
    lista_youtubers = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer']
    
    for yt in lista_youtubers:
        plotar_sensibilidade_toxicidade(yt)