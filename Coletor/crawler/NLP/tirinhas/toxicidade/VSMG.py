import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

YOUTUBERS_LIST = ['Julia MineGirl', 'Tex HS']
TOXICITY_THRESHOLD = 0.7  
BASE_DATA_FOLDER = Path('files')
OUTPUT_FOLDER = Path('output')
SENTIMENT_FILENAME = 'tiras_video.csv'
TOXICITY_FILENAME = 'tiras_video_toxicidade.csv'
BEHAVIORAL_STATES = ['POS-NT', 'NEU-NT', 'NEG-NT', 'POS-T', 'NEU-T', 'NEG-T']

'''
    Função para processar todos os vídeos de um youtuber, criar os estados comportamentais e salvar a sequência completa em um arquivo
    @param youtuber_name - Nome do youtuber a ser analisado
    @return df_video - DataFrame concatenado com as informações de sentimento e toxicidade
'''
def preparar_dados_youtuber(youtuber_name: str) -> pd.DataFrame:
    console.print(f"\n[bold blue]>>>> Preparando dados para: {youtuber_name}[/bold blue]")

    youtuber_base_path = BASE_DATA_FOLDER / youtuber_name
    
    # Lista para guardar os DataFrames de cada vídeo processado
    lista_dfs_videos = []

    if not youtuber_base_path.is_dir():
        console.print(f"[yellow]Aviso: Diretório não encontrado para {youtuber_name}. Pulando.[/yellow]")
        return pd.DataFrame() # Retorna DataFrame vazio em caso de erro

    # Encontrar todos os arquivos de toxicidade
    for toxicity_file_path in youtuber_base_path.rglob(TOXICITY_FILENAME):
        video_folder = toxicity_file_path.parent
        console.print(f"  -> Processando vídeo: {video_folder.name}")

        # Definir o caminho para o arquivo da análise de sentimento
        path_sentimento = video_folder / SENTIMENT_FILENAME

        # Testar se os dados de sentimento existem
        if not path_sentimento.exists():
            console.print(f"     [red]Arquivo de sentimento não encontrado. Pulando vídeo.[/red]")
            continue
            
        # Ler os arquivos de sentimento e de toxicidade
        df_sent = pd.read_csv(path_sentimento)
        df_toxi = pd.read_csv(toxicity_file_path)

        # Unir os DataFrames de forma segura, evitando colunas duplicadas
        df_video = pd.concat([df_sent, df_toxi.drop(columns=df_sent.columns, errors='ignore')], axis=1)
        
        # Criar os estados comportamentais
        if not df_video.empty:
            # Assegurar que as colunas necessárias existem antes de prosseguir
            required_cols = ['toxicity', 'grupo', 'positividade', 'neutralidade', 'negatividade']
            if not all(col in df_video.columns for col in required_cols):
                console.print(f"     [red]Colunas necessárias não encontradas em {video_folder.name}. Pulando.[/red]")
                continue

            conditions = [
                (df_video['toxicity'] < TOXICITY_THRESHOLD) & (df_video['grupo'] == 'POS'),
                (df_video['toxicity'] < TOXICITY_THRESHOLD) & (df_video['grupo'] == 'NEU'),
                (df_video['toxicity'] < TOXICITY_THRESHOLD) & (df_video['grupo'] == 'NEG'),
                (df_video['toxicity'] >= TOXICITY_THRESHOLD) & (df_video['grupo'] == 'POS'),
                (df_video['toxicity'] >= TOXICITY_THRESHOLD) & (df_video['grupo'] == 'NEU'),
                (df_video['toxicity'] >= TOXICITY_THRESHOLD) & (df_video['grupo'] == 'NEG')
            ]
            choices = ['POS-NT', 'NEU-NT', 'NEG-NT', 'POS-T', 'NEU-T', 'NEG-T']
            df_video['estado_comportamental'] = np.select(conditions, choices, default='Indefinido')
            
            # Adicionar o DataFrame processado do vídeo à lista
            lista_dfs_videos.append(df_video)

    if not lista_dfs_videos:
        console.print(f"[yellow]Nenhum vídeo com dados completos foi encontrado para {youtuber_name}.[/yellow]")
        return pd.DataFrame()

    # Concatenar os dados de todos os vídeos em um único DataFrame
    df_youtuber_completo = pd.concat(lista_dfs_videos, ignore_index=True)
    
    # Salvar a sequência completa em um arquivo CSV simples de uma coluna
    all_video_states = df_youtuber_completo['estado_comportamental'].tolist()
    df_sequencia = pd.DataFrame(all_video_states, columns=['estado'])

    output_path = BASE_DATA_FOLDER / youtuber_name / 'sentimento' / 'sequencia_estados.csv'
    df_sequencia.to_csv(output_path, index=False)

    console.print(f"Sequência de {len(all_video_states)} estados para [cyan]{youtuber_name}[/cyan] salva em: [green]{output_path}[/green]") 

    return df_youtuber_completo

'''
    Função para ler um arquivo de sequência de estados e calcular a matriz de transição 6x6
    @param sequencia_path - Caminho para o arquivo de sequência
'''
def calcular_matriz_de_sequencia(sequencia_path: Path) -> pd.DataFrame:
    # Testar se o arquivo de sequência de estados existe
    if not sequencia_path.exists():
        console.print(f"[red]Arquivo de sequência não encontrado: {sequencia_path}[/red]")
        return None

    # Ler o DataFrame de sequência de estados
    df_sequencia = pd.read_csv(sequencia_path)

    # Ler os estados como uma lista
    states_sequence = df_sequencia['estado'].tolist()
    
    # Criar matriz de transição de estados 6x6
    num_states = len(BEHAVIORAL_STATES)
    matrix = np.zeros((num_states, num_states))
    state_to_idx = {label: i for i, label in enumerate(BEHAVIORAL_STATES)}

    # Preencher a matriz com os dados da sequência de transição
    for i in range(len(states_sequence) - 1):
        current_state = states_sequence[i]
        next_state = states_sequence[i+1]
        if current_state in state_to_idx and next_state in state_to_idx:
            matrix[state_to_idx[current_state], state_to_idx[next_state]] += 1
            
    row_sums = matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
    
    df_matrix = pd.DataFrame(prob_matrix, index=BEHAVIORAL_STATES, columns=BEHAVIORAL_STATES)
    return df_matrix

'''
    Função para carregar as matrizes de transição de múltiplos youtubers e gerar um gráfico comparativo com heatmaps lado a lado
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def plotar_comparacao_heatmaps(youtubers_list: list):
    console.print(f"[bold]Iniciando Passo 3: Geração dos Heatmaps Comparativos[/bold]")
    
    num_youtubers = len(youtubers_list)

    # Cria uma figura com subplots: 1 linha, N colunas (uma para cada youtuber)
    fig, axes = plt.subplots(1, num_youtubers, figsize=(12 * num_youtubers, 9), sharey=True)
    
    # Garantir que 'axes' seja sempre uma lista, mesmo com um só youtuber
    if num_youtubers == 1:
        axes = [axes]

    fig.suptitle('Matriz de Transição Comportamental (VSMG de 6 Estados)', fontsize=24, weight='bold')

    for i, youtuber in enumerate(youtubers_list):
        ax = axes[i]
        console.print(f"  -> Gerando heatmap para [cyan]{youtuber}[/cyan]...")
        
        # Carrega o arquivo CSV da matriz
        matrix_path = BASE_DATA_FOLDER / youtuber / 'sentimento' / 'VSMG_6_estados.csv'

        if not matrix_path.exists():
            console.print(f"    [red]Arquivo de matriz não encontrado para {youtuber}. Pulando.[/red]")
            ax.set_title(f"{youtuber}\n(Dados não encontrados)", fontsize=16)
            ax.axis('off') # Esconde o subplot vazio
            continue
            
        df_matrix = pd.read_csv(matrix_path, index_col=0)
        
        # Gera o heatmap com Seaborn
        sns.heatmap(
            df_matrix,
            ax=ax,
            annot=True,          # Mostra os números dentro das células
            fmt=".1%",           # Formata os números como porcentagem (ex: 24.9%)
            cmap='viridis',      # Paleta de cores (outras opções: 'Blues', 'Reds', 'coolwarm')
            linewidths=.5,       # Linhas finas separando as células
            cbar= (i == num_youtubers - 1) # Mostra a barra de cores apenas no último gráfico
        )
        
        ax.set_title(youtuber, fontsize=18, weight='bold')
        ax.set_xlabel('Estado de Destino', fontsize=12)
        if i == 0:
            ax.set_ylabel('Estado de Origem', fontsize=12)
        
    # Ajustar o layout para evitar sobreposição
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta para o título principal caber

    # Salva a figura em um arquivo de imagem
    output_image_path = BASE_DATA_FOLDER / 'sentimento'    
    output_image_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_image_path}/comparativo_vsmg_heatmaps.png')
    console.print(f"\nGráfico comparativo salvo com sucesso em: [green]{output_image_path}[/green]")
    plt.close()

'''
    Função para encontrar e exibir os exemplos de texto mais extremos (maior e menor distância) para cada um dos 6 estados comportamentais
    @param df_completo - DataFrame
'''
def mostrar_exemplos_extremos(df_completo: pd.DataFrame, youtuber_name: str):
    console.print(f"\n--- [bold magenta]Exemplos de Trechos Característicos para: {youtuber_name}[/bold magenta] ---")

    # Definir a lógica de pontuação para encontrar a "maior distância"
    scoring_logic = {
        'POS-NT': df_completo['positividade'] - df_completo['toxicity'],
        'NEU-NT': df_completo['neutralidade'] - df_completo['toxicity'],
        'NEG-NT': df_completo['negatividade'] - df_completo['toxicity'],
        'POS-T':  df_completo['positividade'] + df_completo['toxicity'],
        'NEU-T':  df_completo['neutralidade'] + df_completo['toxicity'],
        'NEG-T':  df_completo['negatividade'] + df_completo['toxicity']
    }

    for estado in BEHAVIORAL_STATES:
        df_estado = df_completo[df_completo['estado_comportamental'] == estado].copy()

        if df_estado.empty:
            console.print(Panel(f"Nenhum exemplo encontrado.", title=f"Estado: [bold]{estado}[/bold]", border_style="yellow"))
            continue

        # Calcular o score para o estado atual
        df_estado['score'] = scoring_logic[estado]
        
        # Encontrar o exemplo de maior distância
        idx_maior = df_estado['score'].idxmax()
        exemplo_maior = df_completo.loc[idx_maior]
        
        texto_maior = f"'[i]{exemplo_maior['tiras']}[/i]'"
        subtitle_maior = (
            f"Pos: {exemplo_maior['positividade']:.1%} | "
            f"Neu: {exemplo_maior['neutralidade']:.1%} | "
            f"Neg: {exemplo_maior['negatividade']:.1%} | "
            f"[b]Tox: {exemplo_maior['toxicity']:.1%}[/b] | "
            f"[b]Sev Tox: {exemplo_maior['severe_toxicity']:.1%}[/b] | "
            f"[b]Sexual: {exemplo_maior['sexual_explicit']:.1%}[/b] | "
        )
        console.print(Panel(texto_maior, title=f"Estado: [bold]{estado}[/bold] (Exemplo de MAIOR Distância)", subtitle=subtitle_maior, border_style="green"))

        # Encontrar o exemplo de menor distância
        # idx_menor = df_estado['score'].idxmin()
        # exemplo_menor = df_completo.loc[idx_menor]

        # texto_menor = f"'[i]{exemplo_menor['tiras']}[/i]'"
        # subtitle_menor = (
        #     f"Pos: {exemplo_menor['positividade']:.1%} | "
        #     f"Neu: {exemplo_menor['neutralidade']:.1%} | "
        #     f"Neg: {exemplo_menor['negatividade']:.1%} | "
        #     f"[b]Tox: {exemplo_menor['toxicity']:.1%}[/b]"
        # )
        # console.print(Panel(texto_menor, title=f"Estado: [bold]{estado}[/bold] (Exemplo de MENOR Distância)", subtitle=subtitle_menor, border_style="yellow"))

'''
    Função para analisar os estados comportamentais de um youtuber, salvar as transições de estados e plotar o mapa de calor dessas transições
'''
def analisar_estados_comportamentais() -> None:
    console.print("[bold]Iniciando Preparação dos Dados de Sequência Comportamental[/bold]")

    for youtuber in YOUTUBERS_LIST:
        preparar_dados_youtuber(youtuber)

    console.print("[bold]Iniciando Cálculo das Matrizes de Transição 6x6[/bold]")


    for youtuber in YOUTUBERS_LIST:
        console.print(f"\n[bold blue]>>>> Calculando matriz para: {youtuber}[/bold blue]")

        input_path = BASE_DATA_FOLDER / youtuber / 'sentimento' / 'sequencia_estados.csv'

        transition_matrix = calcular_matriz_de_sequencia(input_path)

        if transition_matrix is not None:
            output_csv_file = BASE_DATA_FOLDER / youtuber / 'sentimento' / 'VSMG_6_estados.csv'
            transition_matrix.to_csv(output_csv_file)
            console.print(f"Matriz de transição para [cyan]{youtuber}[/cyan] salva em: [green]{output_csv_file}[/green]")

    plotar_comparacao_heatmaps(YOUTUBERS_LIST) 

'''
    Função para identificar exemplos de textos que evidenciem a diferença entre a análise de sentimento e de toxicidade
'''
def analisar_textos_estados_comportamentais() -> None:
    console.print("[bold]Iniciando de Análise de Textos de Estados Comportamentais[/bold]")

    for youtuber in YOUTUBERS_LIST:
        # Chamar a função preparar_dados_youtuber para carregar e processar todos os dados do youtuber
        df_completo_youtuber = preparar_dados_youtuber(youtuber)

        # Verificar se o DataFrame retornado não está vazio antes de prosseguir
        if not df_completo_youtuber.empty:
            mostrar_exemplos_extremos(df_completo_youtuber, youtuber)
        else:
            console.print(f'[yellow]Nenhum dado completo para analisar para {youtuber}[/yellow]')

'''
    Função para encontrar e exibir as N tiras com os maiores ou menores valores para uma métrica específica
    @param youtuber_name - Nome do youtuber a ser analisado
    @param metrica - A métrica do Detoxify a ser analisada (ex: 'toxicity', 'insult', 'obscene')
    @param tipo_ranking - 'maior' para os valores mais altos, 'menor' para os mais baixos
    @param n_exemplos - Quantidade de exemplos a serem mostrados
'''
def encontrar_top_tiras_por_metrica(youtuber_name: str, metrica: str, tipo_ranking: str = 'maior', n_exemplos: int = 5):
    console.print(f"\n[bold]🔎 Buscando Top {n_exemplos} tiras para a métrica '[cyan]{metrica}[/cyan]' ({tipo_ranking.upper()}) de {youtuber_name}...[/bold]")

    # Reutilizar a função de preparação para carregar e agregar todos os dados do youtuber
    df_completo = preparar_dados_youtuber(youtuber_name)
    
    if df_completo.empty:
        console.print(f"[red]Não foi possível carregar dados para {youtuber_name}. Análise cancelada.[/red]")
        return

    # Lista de métricas numéricas disponíveis para análise
    metricas_disponiveis = [col for col in df_completo.columns if pd.api.types.is_numeric_dtype(df_completo[col])]

    # Verificar as métricas
    if metrica not in metricas_disponiveis:
        console.print(f"[red]Erro: Métrica '{metrica}' é inválida ou não foi encontrada.[/red]")
        console.print(f"Métricas numéricas disponíveis: {metricas_disponiveis}")
        return
    
    # Verificar o parâmetro
    if tipo_ranking not in ['maior', 'menor']:
        console.print(f"[red]Erro: Parâmetro 'tipo_ranking' inválido. Use 'maior' ou 'menor'.[/red]")
        return

    # Definir se a ordenação será crescente ou decrescente
    ordem_crescente = True if tipo_ranking == 'menor' else False
    
    df_ordenado = df_completo.sort_values(by=metrica, ascending=ordem_crescente)

    # Selecionar os N melhores resultados
    top_resultados = df_ordenado.head(n_exemplos)

    # Apresentar os resultados em uma tabela formatada
    titulo_tabela = f"Top {n_exemplos} Tiras com '{tipo_ranking.upper()}' Valor para a Métrica '[bold]{metrica}[/bold]'"
    
    tabela = Table(title=titulo_tabela, show_header=True, header_style="bold magenta", expand=True)
    tabela.add_column("Rank", style="dim", width=6)
    tabela.add_column("Tira (Trecho do Texto)", style="italic", min_width=20)
    tabela.add_column(f"Score '{metrica}'", justify="right", style="bold yellow")

    for i, row in enumerate(top_resultados.itertuples(), 1):
        # getattr é uma forma segura de acessar o atributo de um objeto
        texto_tira = getattr(row, 'tiras', 'Texto não encontrado')
        score = getattr(row, metrica, 0.0)
        tabela.add_row(
            f"#{i}",
            texto_tira,
            f"{score:.3%}" # Formata como porcentagem com 3 casas decimais
        )
    
    console.print(tabela)

if __name__ == "__main__":
    #analisar_estados_comportamentais()
    
    #analisar_textos_estados_comportamentais()

    encontrar_top_tiras_por_metrica('Tex HS', 'toxicity', 'maior', 5)
    #encontrar_top_tiras_por_metrica('Tex HS', 'toxicity', 'menor', 5)

    encontrar_top_tiras_por_metrica('Tex HS', 'severe_toxicity', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'severe_toxicity', 'menor', 5)

    # encontrar_top_tiras_por_metrica('Tex HS', 'obscene', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'obscene', 'menor', 5)

    # encontrar_top_tiras_por_metrica('Tex HS', 'identity_attack', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'identity_attack', 'menor', 5)

    # encontrar_top_tiras_por_metrica('Tex HS', 'insult', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'insult', 'menor', 5)

    # encontrar_top_tiras_por_metrica('Tex HS', 'threat', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'threat', 'menor', 5)

    # encontrar_top_tiras_por_metrica('Tex HS', 'sexual_explicit', 'maior', 5)
    # encontrar_top_tiras_por_metrica('Tex HS', 'sexual_explicit', 'menor', 5)