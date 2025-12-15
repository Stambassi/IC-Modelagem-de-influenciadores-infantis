import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
import sys

console = Console()

# Configuração global
BASE_DATA_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'

CONFIG_GERACAO = {
    'toxicidade': {
        'tipo': 'simples',
        'coluna_alvo': 'toxicity',
        'limiares': [0.0, 0.30, 0.70, 1.01],
        'labels': ['NT', 'GZ', 'T']
    },
    'negatividade': {
        'tipo': 'simples',
        'coluna_alvo': 'negatividade',
        'limiares': [0.0, 0.33, 0.66, 1.01],
        'labels': ['BAIXA', 'MEDIA', 'ALTA']
    },
    'misto_9_estados': {
        'tipo': 'combinado',
        'coluna_sentimento': 'sentimento_dominante', 
        'coluna_toxicidade': 'toxicity',
        'limiares_toxicidade': [0.0, 0.30, 0.70, 1.01],
        'labels_toxicidade': ['NT', 'GZ', 'T']
    }
}

'''
    Função central para transformar um DataFrame de tiras em uma lista de estados
    baseado na configuração fornecida

    @param df - DataFrame carregado do tiras_video.csv
    @param config - Dicionário de configuração da análise
'''
def calcular_estados(df: pd.DataFrame, config: dict) -> list:
    # Testa se o tipo de configuração de análise é simples (só Toxicidade ou só Sentimento)
    if config['tipo'] == 'simples':
        # Identifica a coluna alvo
        coluna = config['coluna_alvo']

        # Verifica se a coluna alvo existe no DataFrame
        if coluna not in df.columns:
            raise ValueError(f"Coluna '{coluna}' não encontrada no arquivo.")
            
        # Categoriza a Toxicidade
        return pd.cut(
            df[coluna], 
            bins=config['limiares'], 
            labels=config['labels'], 
            include_lowest=True, 
            right=False
        ).tolist()

    elif config['tipo'] == 'combinado':
        # Identifica as colunas de Toxicidade e Sentimento
        col_sent = config['coluna_sentimento']
        col_tox = config['coluna_toxicidade']

        # Verifica se existem as colunas de Toxicidade e Sentimento no DataFrame        
        if not col_sent or col_tox not in df.columns:
            raise ValueError(f"Colunas necessárias ({col_sent} ou {col_tox}) não encontradas.")

        # Categoriza a Toxicidade temporariamente
        tox_categories = pd.cut(
            df[col_tox],
            bins=config['limiares_toxicidade'],
            labels=config['labels_toxicidade'],
            include_lowest=True,
            right=False
        ).astype(str)

        # Combina Sentimento + Toxicidade (Ex: POS-NT)
        sentimento_series = df[col_sent].astype(str).str.upper()
        
        # Concatenação vetorizada
        estados_combinados = sentimento_series + '-' + tox_categories
        
        return estados_combinados.tolist()
    
    return []

'''
    Função para percorrer os vídeos, calcular a sequência de estados e salvar
    em um arquivo dedicado dentro da pasta do vídeo
    
    @param youtubers_list - Lista de youtubers alvo
    @param tipo_analise - Chave da configuração ('toxicidade', 'negatividade', 'misto_9_estados')
'''
def gerar_arquivos_sequencia(youtubers_list: list[str], tipo_analise: str):
    # Verificar se o tipo de análise existe na configuração
    if tipo_analise not in CONFIG_GERACAO:
        console.print(f"[bold red]Erro: Tipo de análise '{tipo_analise}' não configurado.[/bold red]")
        return

    # Identifica a configuração e cria o nome do arquivo de saída
    config = CONFIG_GERACAO[tipo_analise]
    nome_arquivo_saida = f"sequencia_{tipo_analise}.csv"

    console.print(f"\n[bold magenta]=== Gerando arquivos de sequência: {tipo_analise.upper()} ===[/bold magenta]")

    # Percorre os youtubers
    for youtuber in youtubers_list:
        # Identifica a pasta do youtuber e verifica a sua validade
        base_path = BASE_DATA_FOLDER / youtuber
        if not base_path.is_dir():
            continue
            
        console.print(f"Processando: [cyan]{youtuber}[/cyan]")
        count_processed = 0

        # Busca todos os arquivos de tiras
        for input_path in base_path.rglob(INPUT_FILENAME):
            video_folder = input_path.parent
            
            # Define o caminho da pasta 'sequencias' dentro da pasta do vídeo
            seq_folder = video_folder / 'sequencias'
            output_path = seq_folder / nome_arquivo_saida

            try:
                # Carrega dados
                df = pd.read_csv(input_path)
                if df.empty: 
                    continue

                # Calcula a sequência
                lista_estados = calcular_estados(df, config)
                
                if not lista_estados:
                    continue

                # Cria a pasta se não existir
                seq_folder.mkdir(parents=True, exist_ok=True)

                # Salva a sequência
                df_seq = pd.DataFrame(lista_estados, columns=['estado'])
                df_seq.to_csv(output_path, index=False)
                
                count_processed += 1

            except ValueError as ve:
                # Erros de coluna faltando são esperados em dados incompletos
                console.print(f"  [dim]Pulado {video_folder.name}: {ve}[/dim]")
                pass
            except Exception as e:
                console.print(f"  [red]Erro em {video_folder.name}: {e}[/red]")

        console.print(f"  -> Total de vídeos processados: {count_processed}")

if __name__ == "__main__":
    lista_youtubers = ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'Papile', 'Robin Hood Gamer', 'TazerCraft', 'Tex HS']

    gerar_arquivos_sequencia(lista_youtubers, 'toxicidade')
    #gerar_arquivos_sequencia(lista_youtubers, 'sentimento')
    #gerar_arquivos_sequencia(lista_youtubers, 'misto_9_estados')