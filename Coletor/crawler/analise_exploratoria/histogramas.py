import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

BASE_FOLDER = Path('files')

'''
    Função genérica para gerar e salvar um histograma de QUALQUER coluna numérica, 
    buscando em qualquer um dos arquivos de dados padrão
    
    @param youtubers_list - Lista de youtubers a serem analisados
    @param coluna_alvo - O nome da coluna numérica no CSV (ex: 'toxicity', 'duration', 'views')
    @param arquivo_alvo - O nome do arquivo onde a coluna está (ex: 'tiras_video.csv', 'videos_info.csv')
    @param bins - Número de divisões do histograma
    @param log_scale - Se True, usa escala logarítmica no eixo Y
    @param titulo_grafico - (Opcional) Título customizado. Se None, gera automático.
    @param limiares - (Opcional) Lista de valores para desenhar linhas verticais (ex: [0.30, 0.70]).
'''
def plotar_histograma_generico(
    youtubers_list: list[str], 
    coluna_alvo: str, 
    arquivo_alvo: str, 
    bins: int = 50, 
    log_scale: bool = False,
    titulo_grafico: str = None,
    limiares: list[float] = None
):
    titulo_padrao = f"Distribuição de '{coluna_alvo}' ({arquivo_alvo})"
    titulo_uso = titulo_grafico if titulo_grafico else titulo_padrao
    
    console.print(f"\n--- [bold magenta]{titulo_uso}[/bold magenta] ---")
    
    todos_valores = []

    # Coleta de Dados
    for youtuber in youtubers_list:
        youtuber_path = BASE_FOLDER / youtuber
        if not youtuber_path.is_dir():
            continue
            
        console.print(f"Coletando dados de: [cyan]{youtuber}[/cyan]...")
        
        # Busca recursiva pelo arquivo especificado
        for arquivo_path in youtuber_path.rglob(arquivo_alvo):
            try:
                # Ler o arquivo especificado
                df = pd.read_csv(arquivo_path)
                if coluna_alvo in df.columns:
                    # Conversão específica para coluna específica
                    if coluna_alvo == 'duration':
                        # Converte formato ISO 8601 (PT1H2M) para Timedelta e depois para Segundos Totais
                        # errors='coerce' transforma valores inválidos em NaT (Not a Time), que depois são removidos
                        valores = pd.to_timedelta(df[coluna_alvo], errors='coerce').dt.total_seconds().dropna().tolist()
                        
                        # Converter para minutos
                        valores = [v / 60 for v in valores] 
                    else:
                        # Lógica padrão para números (toxicity, views_count, likes_count)
                        valores = pd.to_numeric(df[coluna_alvo], errors='coerce').dropna().tolist()

                    todos_valores.extend(valores)
            except Exception as e:
                console.print(f"  [red]Erro ao ler {arquivo_path.name}: {e}[/red]")

    if not todos_valores:
        console.print(f"[bold red]Nenhum dado encontrado para a coluna '{coluna_alvo}' em '{arquivo_alvo}'.[/bold red]")
        return

    # Preparação para plotagem
    console.print(f"Total de amostras coletadas: {len(todos_valores)}")
    
    plt.figure(figsize=(10, 6))
    
    # Define a cor baseada no tipo de arquivo para diferenciação visual rápida
    cor_barras = '#4C72B0' if 'tiras' in arquivo_alvo else '#55A868' # Azul para tiras, Verde para vídeos
    label_y = 'Frequência (Nº de Segmentos)' if 'tiras' in arquivo_alvo else 'Frequência (Nº de Vídeos)'
    
    if log_scale: label_y += ' [Log]'

    # Cria o histograma
    n, bins_edges, patches = plt.hist(
        todos_valores, 
        bins=bins, 
        color=cor_barras, 
        edgecolor='black', 
        alpha=0.7,
        log=log_scale
    )

    # Adicionar linhas de limiar se existirem
    if limiares:
        cores_limiar = ['orange', 'red', 'purple', 'brown'] # Paleta para múltiplos limiares

        for i, valor in enumerate(limiares):
            cor = cores_limiar[i % len(cores_limiar)]
            plt.axvline(valor, color=cor, linestyle='--', linewidth=1.5, label=f'Limiar {i+1} ({valor})')

        plt.legend()

    # Customização
    plt.title(f"{titulo_uso}\n(Baseado em {len(todos_valores)} amostras)", fontsize=14)
    plt.xlabel(f"Valor de '{coluna_alvo}'", fontsize=12)
    plt.ylabel(label_y, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Ajuste inteligente do eixo X para não ficar "colado"
    min_val, max_val = min(todos_valores), max(todos_valores)
    margem = (max_val - min_val) * 0.05 if max_val != min_val else 0.1

    # Se os dados forem normalizados (0 a 1), fixa em 0-1, senão usa o range dos dados
    if min_val >= 0 and max_val <= 1.05: 
        plt.xlim(0, 1)
    else:
        plt.xlim(min_val - margem, max_val + margem)

    # Salva o arquivo
    output_dir = BASE_FOLDER / 'analise_exploratoria'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Nome do arquivo limpo e seguro
    nome_safe = f"histograma_{arquivo_alvo.split('.')[0]}_{coluna_alvo}.png"
    output_path = output_dir / nome_safe
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Histograma salvo em: {output_path}[/green]")
    
    # Estatísticas básicas
    console.print(f"Estatísticas de '{coluna_alvo}':")
    console.print(f"  Média: {np.mean(todos_valores):.4f}")
    console.print(f"  Mediana: {np.median(todos_valores):.4f}")
    console.print(f"  Mínimo: {np.min(todos_valores):.4f} | Máximo: {np.max(todos_valores):.4f}")


if __name__ == "__main__":
    lista_youtubers = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer']

    # Toxicidade (tiras)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='toxicity',
        arquivo_alvo='tiras_video.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição Global de Toxicidade",
        limiares=[0.30, 0.70] # Grey Zone e Toxicidade
    )

    # Negatividade (tiras)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='negatividade',
        arquivo_alvo='tiras_video.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição Global de Negatividade"
    )

    # Neutralidade (tiras)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='neutralidade',
        arquivo_alvo='tiras_video.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição Global de Neutralidade"
    )

    # Positividade (tiras)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='positividade',
        arquivo_alvo='tiras_video.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição Global de Positividade"
    )

    # Duração dos vídeos (info do vídeo)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='duration', 
        arquivo_alvo='videos_info.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição de Duração dos Vídeos (Minutos)"
    )

    # Visualizações (info do vídeo)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='view_count',
        arquivo_alvo='videos_info.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição de Visualizações"
    )

    # Comentários (info do vídeo)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='comment_count',
        arquivo_alvo='videos_info.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição de Comentários"
    )

    # Likes (info do vídeo)
    plotar_histograma_generico(
        youtubers_list=lista_youtubers,
        coluna_alvo='like_count',
        arquivo_alvo='videos_info.csv',
        bins=100,
        log_scale=False,
        titulo_grafico="Distribuição de Curtidas"
    )