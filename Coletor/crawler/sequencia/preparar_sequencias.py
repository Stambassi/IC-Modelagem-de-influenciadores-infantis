import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

# Configuração global
BASE_DATA_FOLDER = Path('files')
OUTPUT_ROOT_FOLDER = BASE_DATA_FOLDER / 'sequencias'

# Mapeia cada youtuber para sua categoria principal (Minecraft, Roblox, etc.)
MAPA_YOUTUBERS_CATEGORIA = {
    'Julia MineGirl': 'Roblox',
    #'Tex HS': 'Misto',
    'Robin Hood Gamer': 'Minecraft'
}

# Configurações de análise (limiares e estados)
CONFIG_ANALISE = {
    'toxicidade': {
        'tipo': 'simples', 
        'coluna_alvo': 'toxicity',
        'limiares': [0.0, 0.30, 0.70, 1.01],
        'labels': ['NT', 'GZ', 'T'],
        'eventos_gatilho': ['T']
    },
    'negatividade': {
        'tipo': 'simples',
        'coluna_alvo': 'negatividade', 
        'limiares': [0.0, 0.33, 0.66, 1.01], 
        'labels': ['BAIXA_NEG', 'MEDIA_NEG', 'ALTA_NEG'],
        'eventos_gatilho': ['ALTA_NEG']
    },
    'misto_9_estados': {
        'tipo': 'combinado', 
        'coluna_sentimento': 'sentimento_dominante', # Coluna categórica (POS, NEU, NEG)
        'coluna_toxicidade': 'toxicity', # Coluna numérica (0.0 - 1.0)
        'limiares_toxicidade': [0.0, 0.30, 0.70, 1.01], 
        'labels_toxicidade': ['NT', 'GZ', 'T'],
        'eventos_gatilho': ['POS-T', 'NEU-T', 'NEG-T'] 
    }
}

'''
    Função para varrer as pastas de um youtuber, ler os arquivos brutos 'tiras_video.csv'
    e construir a lista de sequências de todos eles
    
    @param youtuber_name - Nome do youtuber
    @param config - Dicionário de configuração da análise
    @return list[list[str]] - Lista onde cada item é a sequência de estados de um vídeo
'''
def carregar_sequencias_videos(youtuber_name: str, config: dict) -> list[list[str]]:
    # Identifica a pasta do youtuber e verifica se ela existe
    youtuber_path = BASE_DATA_FOLDER / youtuber_name
    if not youtuber_path.is_dir():
        console.print(f"     [red]Diretório não encontrado: {youtuber_path}[/red]")
        return []

    # Define as variáveis iniciais
    todas_sequencias_videos = []

    # Ordena a lista de arquivos de tiras
    arquivos_csv = sorted(list(youtuber_path.rglob('tiras_video.csv')))

    # Verifica que a lista de tiras existe
    if not arquivos_csv:
        return []

    # Itera sobre cada arquivo de tira
    for csv_path in arquivos_csv:
        try:
            # Lê o arquivo de tira
            df_tira = pd.read_csv(csv_path)
            
            # Verifica se o arquivo de tira não está vazio
            if df_tira.empty:
                continue

            estados_video = []

            # Análise simples (Ex: só toxicidade ou só sentimento)
            if config.get('tipo') == 'simples':
                coluna = str(config['coluna_alvo'])
                
                # Verificação de segurança específica para análise simples
                if coluna not in df_tira.columns: 
                    continue

                estados_video = pd.cut(
                    df_tira[coluna], 
                    bins=config['limiares'], 
                    labels=config['labels'], 
                    include_lowest=True, 
                    right=False
                ).tolist()

            # Análise combinada (9 estados: sentimento + toxicidade)
            elif config.get('tipo') == 'combinado':
                col_sent = config['coluna_sentimento']
                col_tox = config['coluna_toxicidade']

                # Verificação de segurança
                if not all(col in df_tira.columns for col in [col_sent, col_tox]):
                    continue
                
                # Extração dos limiares da lista [0.0, 0.30, 0.70, 1.01]
                limiares = config['limiares_toxicidade']
                low = limiares[1]  # 0.30
                high = limiares[2] # 0.70
                
                # Lógica de classificação dos 9 estados
                conditions = [
                    # NT (Não-Tóxico): Abaixo de 0.30
                    (df_tira[col_tox] < low) & (df_tira[col_sent] == 'POS'),
                    (df_tira[col_tox] < low) & (df_tira[col_sent] == 'NEU'),
                    (df_tira[col_tox] < low) & (df_tira[col_sent] == 'NEG'),
                    
                    # GZ (Zona Cinza): Entre 0.30 (inclusive) e 0.70 (exclusive)
                    (df_tira[col_tox] >= low) & (df_tira[col_tox] < high) & (df_tira[col_sent] == 'POS'),
                    (df_tira[col_tox] >= low) & (df_tira[col_tox] < high) & (df_tira[col_sent] == 'NEU'),
                    (df_tira[col_tox] >= low) & (df_tira[col_tox] < high) & (df_tira[col_sent] == 'NEG'),
                    
                    # T (Tóxico): Acima de 0.70 (inclusive)
                    (df_tira[col_tox] >= high) & (df_tira[col_sent] == 'POS'),
                    (df_tira[col_tox] >= high) & (df_tira[col_sent] == 'NEU'),
                    (df_tira[col_tox] >= high) & (df_tira[col_sent] == 'NEG')
                ]
                
                choices = ['POS-NT', 'NEU-NT', 'NEG-NT', 'POS-GZ', 'NEU-GZ', 'NEG-GZ', 'POS-T', 'NEU-T', 'NEG-T']
                
                estados_video = np.select(conditions, choices, default='Indefinido').tolist()
            
            # Se existir, adiciona a lista do vídeo inteiro como um item na lista principal
            if estados_video:
                todas_sequencias_videos.append(estados_video)
            
        except Exception as e:
            console.print(f"     [red]Erro ao ler {csv_path.name}: {e}[/red]")

    return todas_sequencias_videos

'''
    Função para extrair as subsequências que antecedem um evento de interesse
    
    @param lista_de_videos - Lista de listas (cada item é um vídeo)
    @param eventos_gatilho - Lista de estados que disparam a extração (ex: ['T'])
    @param tamanho_janela - Quantos estados anteriores extrair (ex: 3)
    @return list[list[str]] - Banco de dados de sequências precursoras
'''
def extrair_sequencias_precursoras(lista_de_videos: list[list[str]], eventos_gatilho: list, tamanho_janela: int) -> list[list[str]]:
    dataset_precursor = []
    
    # Iterar sobre cada vídeo individualmente
    for sequencia_video in lista_de_videos:
        # Iterar pelos estados dentro deste vídeo
        for i in range(tamanho_janela, len(sequencia_video)):
            estado_atual = sequencia_video[i]
            
            if estado_atual in eventos_gatilho:
                # Extrai os 'k' estados anteriores deste vídeo
                janela_anterior = sequencia_video[i - tamanho_janela : i]
                
                if len(janela_anterior) == tamanho_janela:
                    dataset_precursor.append(janela_anterior)
                
    return dataset_precursor

'''
    Função para salvar as sequências extraídas em um arquivo CSV simples na pasta correta.
    
    @param sequencias - A lista de listas de estados.
    @param pasta_grupo - O nome da subpasta onde salvar (ex: 'Minecraft', 'Geral').
    @param nome_arquivo - O nome do arquivo CSV.
'''
def salvar_sequencias_para_mineracao(sequencias: list[list[str]], pasta_grupo: str, nome_arquivo: str) -> None:
    if not sequencias:
        console.print(f"     [yellow]Nenhuma sequência encontrada para o grupo '{pasta_grupo}'.[/yellow]")
        return

    try:
        # Definir o caminho de saída: files/sequencias/{Grupo}/arquivo.csv
        path_saida_dir = OUTPUT_ROOT_FOLDER / pasta_grupo
        path_saida_dir.mkdir(parents=True, exist_ok=True)
        
        path_saida_arquivo = path_saida_dir / nome_arquivo

        # Converter para DataFrame e salvar sem cabeçalho
        df_export = pd.DataFrame(sequencias)
        df_export.to_csv(path_saida_arquivo, index=False, header=False)
        
        console.print(f"     [green]Dataset salvo:[/green] {path_saida_arquivo} ({len(sequencias)} sequências)")
        
    except Exception as e:
        console.print(f"     [red]Erro ao salvar arquivo: {e}[/red]")

'''
    Função auxiliar para processar um grupo específico de youtubers (Individual, Categoria ou Geral)
    
    @param nome_grupo - Nome do grupo (ex: 'Geral', 'Minecraft', 'Julia MineGirl')
    @param lista_youtubers - Lista de nomes dos youtubers que compõem este grupo
    @param config - Configuração da análise
    @param tamanho_janela - Tamanho da janela de sequência
    @param tipo_analise - Nome da análise para o arquivo
'''
def processar_grupo(nome_grupo: str, lista_youtubers: list, config: dict, tamanho_janela: int, tipo_analise: str):
    console.print(f"\n  [bold cyan]Processando Grupo: {nome_grupo}[/bold cyan] (Youtubers: {len(lista_youtubers)})")
    
    # Esta lista conterá todos os vídeos de todos os youtubers do grupo
    todos_videos_do_grupo = []

    # Carrega dados de todos os membros
    for youtuber in lista_youtubers:
        videos_youtuber = carregar_sequencias_videos(youtuber, config)
        todos_videos_do_grupo.extend(videos_youtuber) # Adiciona a lista de vídeos deste youtuber
    
    if not todos_videos_do_grupo:
        console.print(f"     [yellow]Nenhum dado encontrado para o grupo {nome_grupo}.[/yellow]")
        return

    # Extrai aequências
    sequencias_precursoras = extrair_sequencias_precursoras(
        todos_videos_do_grupo, 
        config['eventos_gatilho'], 
        tamanho_janela
    )

    # Salva os dados
    nome_arquivo = f"sequencias_{tipo_analise}_{nome_grupo.replace(' ', '_')}.csv"
    salvar_sequencias_para_mineracao(sequencias_precursoras, nome_grupo, nome_arquivo)

'''
    Função principal para orquestrar a preparação dos dados em múltiplas dimensões
'''
def orquestrar_preparacao_sequencias_multidimensional(tipo_analise: str, tamanho_janela: int):
    if tipo_analise not in CONFIG_ANALISE:
        console.print(f"[bold red]Erro: Tipo de análise '{tipo_analise}' não configurado.[/bold red]")
        return
    
    config = CONFIG_ANALISE[tipo_analise]
    
    console.print(f"\n[bold magenta]=== Preparação de Sequências Multidimensional ({tipo_analise.upper()}) ===[/bold magenta]")
    console.print(f"Janela: {tamanho_janela} estados anteriores ao gatilho {config['eventos_gatilho']}")

    '''
    # Análise da dimensão individual
    console.print("\n[bold]1. Dimensão: Individual (Por Youtuber)[/bold]")
    for youtuber in MAPA_YOUTUBERS_CATEGORIA.keys():
        processar_grupo(youtuber, [youtuber], config, tamanho_janela, tipo_analise)

    # Análise da dimensão de categoria (Minecraft vs Roblox)
    console.print("\n[bold]2. Dimensão: Categoria (Por Jogo)[/bold]")

    # Inverte o mapa para agrupar por categoria: {'Roblox': ['Julia', 'Tex'], ...}
    categorias = {}
    for youtuber, cat in MAPA_YOUTUBERS_CATEGORIA.items():
        if cat not in categorias:
            categorias[cat] = []
        categorias[cat].append(youtuber)
    
    for nome_cat, lista_membros in categorias.items():
        processar_grupo(nome_cat, lista_membros, config, tamanho_janela, tipo_analise)
    '''

    # Análise da dimensão geral (todos)
    console.print("\n[bold]3. Dimensão: Geral (Todos os Dados)[/bold]")
    todos_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    processar_grupo("Geral", todos_youtubers, config, tamanho_janela, tipo_analise)

if __name__ == "__main__":
    JANELA = 3

    orquestrar_preparacao_sequencias_multidimensional('toxicidade', JANELA)
    orquestrar_preparacao_sequencias_multidimensional('misto_9_estados', JANELA)
    orquestrar_preparacao_sequencias_multidimensional('negatividade', JANELA)