import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
import matplotlib.pyplot as plt

console = Console()

# Configuração global
BASE_DATA_FOLDER = Path('files')
OUTPUT_ROOT_FOLDER = BASE_DATA_FOLDER / 'sequencias'

# Mapeia cada youtuber para sua categoria principal (Minecraft, Roblox, etc.)
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
    'Tex HS': 'Misto'
}

#Amy Scarlet', 'AuthenticGames', 'Cadres', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'Papile', 'Robin Hood Gamer', 'TazerCraft, ''Tex HS']

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

# Configurações globais de plotagem
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
            
            # Identificar o ID do vídeo
            df_video = pd.read_csv(csv_path.parent / 'videos_info.csv')
            video_id = df_video['video_id'][0]

            # Se existir, adiciona a lista do vídeo inteiro como um item na lista principal
            if estados_video:
                todas_sequencias_videos.append({
                    'id': video_id,
                    'sequencia': estados_video
                })
            
        except Exception as e:
            console.print(f"     [red]Erro ao ler {csv_path.name}: {e}[/red]")

    return todas_sequencias_videos

'''
    Função para extrair as subsequências que antecedem e sucedem um evento de interesse
    
    @param lista_de_videos - Lista de listas (cada item é um vídeo)
    @param eventos_gatilho - Lista de estados que disparam a extração (ex: ['T'])
    @param tamanho_janela - Quantos estados anteriores extrair (ex: 3)
    @return list[list[str]] - Banco de dados de sequências precursoras
'''
def extrair_sequencias_janela_completa(lista_de_videos: list[dict], eventos_gatilho: list, tamanho_janela: int) -> list[list]:
    dataset_janelas = []
    
    for item in lista_de_videos:
        video_id = item['id']
        sequencia_video = item['sequencia']
        
        # Percorre o vídeo garantindo espaço para a janela anterior e posterior
        for i in range(tamanho_janela, len(sequencia_video) - tamanho_janela):
            estado_atual = sequencia_video[i]
            
            if estado_atual in eventos_gatilho:
                # Extrai anteriores
                anteriores = sequencia_video[i - tamanho_janela : i]
                # Extrai posteriores
                posteriores = sequencia_video[i + 1 : i + 1 + tamanho_janela]
                
                # Monta a linha: [ID, ant..., GATILHO, post...]
                linha_completa = [video_id] + anteriores + [estado_atual] + posteriores
                dataset_janelas.append(linha_completa)
                
    return dataset_janelas

'''
    Função para salvar as sequências extraídas em um arquivo CSV simples na pasta correta.
    
    @param sequencias - A lista de listas de estados.
    @param pasta_grupo - O nome da subpasta onde salvar (ex: 'Minecraft', 'Geral').
    @param nome_arquivo - O nome do arquivo CSV.
'''
def salvar_sequencias_para_mineracao(sequencias: list[list], pasta_grupo: str, nome_arquivo: str, tamanho_janela: int) -> None:
    if not sequencias:
        console.print(f"     [yellow]Nenhuma sequência encontrada para o grupo '{pasta_grupo}'.[/yellow]")
        return

    try:
        path_saida_dir = OUTPUT_ROOT_FOLDER / pasta_grupo
        path_saida_dir.mkdir(parents=True, exist_ok=True)
        path_saida_arquivo = path_saida_dir / nome_arquivo

        # Geração Dinâmica do Cabeçalho
        cols_anteriores = [f"t-{i}" for i in range(tamanho_janela, 0, -1)]
        cols_posteriores = [f"t+{i}" for i in range(1, tamanho_janela + 1)]
        header = ['video_id'] + cols_anteriores + ['evento'] + cols_posteriores

        # Converter para DataFrame
        df_export = pd.DataFrame(sequencias, columns=header)
        
        # Salvar com cabeçalho (header=True)
        df_export.to_csv(path_saida_arquivo, index=False, header=True)
        
        console.print(f"     [green]Dataset salvo:[/green] {path_saida_arquivo} ({len(sequencias)} sequências)")
        
    except Exception as e:
        console.print(f"     [red]Erro ao salvar arquivo: {e}[/red]")

'''
    Função auxiliar para renderizar um DataFrame como uma tabela estática
    
    @param df - O DataFrame com os dados estatísticos
    @param titulo - O título da tabela
    @param caminho_saida - Path para salvar a imagem
'''
def renderizar_tabela_academica(df: pd.DataFrame, titulo: str, caminho_saida: Path):
    if df.empty:
        return

    # Dimensões da figura baseadas no tamanho do dataframe
    # Altura: cabeçalho + linhas + margem
    largura = 12
    altura = len(df) * 0.5 + 1.5 
    
    fig, ax = plt.subplots(figsize=(largura, altura))
    ax.axis('off')

    # Criar a tabela
    tabela = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    # Estilização Fina
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1, 1.8) # Aumenta altura das células

    # Cores
    cor_cabecalho = '#2c3e50' 
    cor_texto_cabecalho = 'white'
    cor_zebra_1 = '#ffffff'
    cor_zebra_2 = '#f8f9fa'
    cor_borda = '#dddddd'

    # Iterar sobre as células para aplicar estilos
    for (row, col), cell in tabela.get_celld().items():
        cell.set_edgecolor(cor_borda)
        cell.set_linewidth(0.5)

        # Cabeçalho
        if row == 0:
            cell.set_text_props(weight='bold', color=cor_texto_cabecalho)
            cell.set_facecolor(cor_cabecalho)
            cell.set_fontsize(12)
        # Dados
        else:
            # Alinhamento específico: Primeira coluna (Transição) à esquerda
            if col == 0:
                cell.set_text_props(ha='left')
            
            # Zebra Striping (Linhas alternadas)
            if row % 2 == 0:
                cell.set_facecolor(cor_zebra_2)
            else:
                cell.set_facecolor(cor_zebra_1)

    # Adicionar Título
    plt.title(titulo, pad=20, fontsize=14, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    
    try:
        plt.savefig(caminho_saida, bbox_inches='tight', pad_inches=0.2)
        console.print(f"     [green]Tabela salva com sucesso:[/green] {caminho_saida}")
    except Exception as e:
        console.print(f"     [red]Erro ao salvar tabela: {e}[/red]")
    finally:
        plt.close()

'''
    Função para calcular estatísticas de Inter Arrival Time of Toxicity (IATT) entre estados.
    Gera um DataFrame e invoca o renderizador gráfico.
    
    @param lista_de_videos - Lista de dicionários contendo {'id': str, 'sequencia': list}
    @param estados_interesse - Lista de estados para analisar (ex: ['T', 'NT', 'GZ'])
    @param nome_grupo - Nome do grupo sendo analisado (para o título e arquivo)
'''
def calcular_e_plotar_estatisticas_iatt(lista_de_videos: list[dict], estados_interesse: list, nome_grupo: str):
    # Dicionário para armazenar as listas de tempos: delays[(Origem, Destino)] = [tempo1, tempo2...]
    delays = {}
    
    # Inicializa as chaves possíveis
    for origem in estados_interesse:
        for destino in estados_interesse:
            delays[(origem, destino)] = []

    # Processamento dos dados
    for item in lista_de_videos: 
        # Proteção para garantir que estamos acessando a lista correta
        if isinstance(item, dict) and 'sequencia' in item:
            video = item['sequencia']
        elif isinstance(item, list):
            video = item
        else:
            continue
        
        last_seen_index = {estado: -1 for estado in estados_interesse}

        for current_idx, current_state in enumerate(video):
            if current_state not in estados_interesse:
                continue

            for prev_state in estados_interesse:
                idx_prev = last_seen_index[prev_state]
                
                if idx_prev != -1:
                    distancia = current_idx - idx_prev
                    if distancia > 0:
                        delays[(prev_state, current_state)].append(distancia)

            last_seen_index[current_state] = current_idx

    # Preparação dos dados para a Tabela
    linhas_tabela = []

    for origem in estados_interesse:
        for destino in estados_interesse:
            data = delays.get((origem, destino), [])
            
            # Filtro opcional: Só mostrar se tiver relevância estatística (> 5 ocorrências)
            # Para mostrar tudo, remova o if.
            if len(data) > 0:
                arr = np.array(data)
                
                # Cálculo das métricas
                media = np.mean(arr)
                mediana = np.median(arr)
                std_dev = np.std(arr)
                min_val = np.min(arr)
                max_val = np.max(arr)
                cv = (std_dev / media) if media > 0 else 0 # Coeficiente de Variação
                
                # Formatação das strings para a tabela
                linhas_tabela.append({
                    "Transição": f"{origem} $\\rightarrow$ {destino}",
                    "Quantidade": f"{len(arr)}",
                    "Média": f"{media:.2f}",
                    "Mediana": f"{mediana:.1f}",
                    "Desvio Padrão": f"{std_dev:.2f}",
                    "CV": f"{cv:.2f}",
                    "Range (Min-Max)": f"{min_val} - {max_val}"
                })
    
    # Criar DataFrame
    df_stats = pd.DataFrame(linhas_tabela)
    
    if df_stats.empty:
        console.print(f"[yellow]Sem dados de transição suficientes para IATT em {nome_grupo}.[/yellow]")
        return

    # Definir caminho de saída
    output_dir = Path(f'files/sequencias/{nome_grupo}')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'tabela_IATT_{nome_grupo}.png'

    # Renderizar
    titulo_tabela = f"Estatísticas de Tempo de Chegada (IATT) - Grupo: {nome_grupo}"
    renderizar_tabela_academica(df_stats, titulo_tabela, output_file)

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
    sequencias = extrair_sequencias_janela_completa(
        todos_videos_do_grupo, 
        config['eventos_gatilho'], 
        tamanho_janela
    )

    # Salva os dados
    nome_arquivo = f"sequencias_{tipo_analise}_{nome_grupo.replace(' ', '_')}.csv"
    salvar_sequencias_para_mineracao(sequencias, nome_grupo, nome_arquivo, tamanho_janela)

    # Cálculo de IATT (Estatísticas de tempo entre estados)
    labels_interesse = []
    if 'labels' in config:
        labels_interesse = config['labels'] # Caso simples (NT, GZ, T)
    elif 'labels_toxicidade' in config:
        # Caso combinado, usamos os 9 estados gerados
        labels_interesse = ['POS-NT', 'NEU-NT', 'NEG-NT', 'POS-GZ', 'NEU-GZ', 'NEG-GZ', 'POS-T', 'NEU-T', 'NEG-T']

    if labels_interesse:
        calcular_e_plotar_estatisticas_iatt(todos_videos_do_grupo, labels_interesse, nome_grupo)

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

    # Análise da dimensão geral (todos)
    console.print("\n[bold]3. Dimensão: Geral (Todos os Dados)[/bold]")
    todos_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    processar_grupo("Geral", todos_youtubers, config, tamanho_janela, tipo_analise)

if __name__ == "__main__":
    JANELA = 4

    orquestrar_preparacao_sequencias_multidimensional('toxicidade', JANELA)
    #orquestrar_preparacao_sequencias_multidimensional('misto_9_estados', JANELA)
    #orquestrar_preparacao_sequencias_multidimensional('negatividade', JANELA)