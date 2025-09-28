import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console

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
'''
def preparar_dados_youtuber(youtuber_name: str) -> None:
    console.print(f"\n[bold blue]>>>> Preparando dados para: {youtuber_name}[/bold blue]")

    youtuber_base_path = BASE_DATA_FOLDER / youtuber_name
    all_video_states = []

    if not youtuber_base_path.is_dir():
        console.print(f"[yellow]Aviso: Diretório não encontrado para {youtuber_name}. Pulando.[/yellow]")
        return

    # Encontrar todos os arquivos de toxicidade
    for toxicity_file_path in youtuber_base_path.rglob(TOXICITY_FILENAME):
        video_folder = toxicity_file_path.parent
        console.print(f"  -> Processando vídeo: {video_folder.name}")

        # Carregar e unir os dados de sentimento e toxicidade
        path_sentimento = video_folder / SENTIMENT_FILENAME
        if not path_sentimento.exists():
            console.print(f"     [red]Arquivo de sentimento não encontrado. Pulando vídeo.[/red]")
            continue
            
        df_sent = pd.read_csv(path_sentimento)
        df_toxi = pd.read_csv(toxicity_file_path)
        colunas_toxicidade = ['toxicity']
        df_video = pd.concat([df_sent, df_toxi[colunas_toxicidade]], axis=1)

        # Criar os estados comportamentais
        if not df_video.empty:
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
            
            # Adicionar a sequência de estados do vídeo à lista geral
            all_video_states.extend(df_video['estado_comportamental'].tolist())

    if not all_video_states:
        console.print(f"[yellow]Nenhuma sequência de estados gerada para {youtuber_name}.[/yellow]")
        return

    # Salvar a sequência completa em um arquivo CSV simples de uma coluna
    df_sequencia = pd.DataFrame(all_video_states, columns=['estado'])
    output_path = BASE_DATA_FOLDER / youtuber_name / 'sentimento' / 'sequencia_estados.csv'
    df_sequencia.to_csv(output_path, index=False)
    
    console.print(f"Sequência de {len(all_video_states)} estados para [cyan]{youtuber_name}[/cyan] salva em: [green]{output_path}[/green]")

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

if __name__ == "__main__":
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