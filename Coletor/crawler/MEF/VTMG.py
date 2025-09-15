from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from pandas.api.types import CategoricalDtype

console = Console()

'''
    Função para armazenar as transições de estados de sentimento (POS, NEU, NEG) 
    de cada vídeo em um arquivo CSV.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_transicoes_sentimento(youtubers_list: list[str]) -> None:
    # Nomes dos estados de sentimento
    estados_sentimento = ['POS', 'NEU', 'NEG']
    # Colunas que espera-se encontrar no CSV de entrada
    colunas_sentimento = ['toxicidade', 'positividade', 'neutralidade']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando transições de sentimento para [bold cyan]{youtuber}[/bold cyan]')

        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            video_path = tiras_csv_path.parent
            
            try:
                (video_path / 'transicoes').mkdir(parents=True, exist_ok=True)
                
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se o DataFrame tem as colunas necessárias e dados suficientes
                if df_tiras_video.empty or not all(col in df_tiras_video.columns for col in colunas_sentimento) or len(df_tiras_video) < 2:
                    continue

                # Usar a coluna 'grupo' como a coluna de 'estado'
                df_tiras_video.rename(columns={'grupo': 'estado'}, inplace=True)
                
                # Para robustez, garantimos que apenas os valores esperados ('POS', 'NEU', 'NEG') sejam considerados
                df_tiras_video = df_tiras_video[df_tiras_video['estado'].isin(estados_sentimento)]
                
                # Re-verificar se ainda há dados suficientes após a filtragem
                if len(df_tiras_video) < 2:
                    continue
                
                # Converter a coluna 'estado' para o tipo Categórico para que o groupby gere todas as 3x3=9 transições
                tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
                df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)

                # Lógica de transição
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)
                
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa e conta, incluindo as transições que não ocorreram (contagem 0)
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')

                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Salva no novo arquivo CSV com nome descritivo
                output_path = video_path / 'transicoes' / 'transicoes_sentimento.csv'
                contagem.to_csv(output_path, index=False)
            
            except Exception as e:
                console.print(f'[bold red]Erro[/bold red] em {video_path.name} (salvar_transicoes_sentimento): {e}')

'''
    Função para criar e persistir a Matriz de Transição de Sentimento (POS, NEU, NEG) para cada vídeo.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_matriz_transicao_sentimento_video(youtubers_list: list[str]) -> None:
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes de sentimento para [bold cyan]{youtuber}[/bold cyan]')

        # Procurar recursivamente pelo arquivo de transições de sentimento
        for transicoes_csv_path in base_path.rglob('transicoes_sentimento.csv'):                
            try:                
                df_transicoes = pd.read_csv(transicoes_csv_path)

                if df_transicoes.empty:
                    continue

                # Para garantir que a matriz final seja sempre 3x3, mesmo que um estado nunca apareça como ponto de partida, converte-se as colunas para o tipo Categórico.
                estados_sentimento = ['POS', 'NEU', 'NEG']
                tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
                df_transicoes['estado'] = df_transicoes['estado'].astype(tipo_categorico)
                df_transicoes['proximo_estado'] = df_transicoes['proximo_estado'].astype(tipo_categorico)

                # Calcular a soma das transições que saem de cada estado
                # # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence. 
                somas_por_estado = df_transicoes.groupby('estado', observed=False)['contagem'].transform('sum')

                # Calcular a probabilidade de cada transição 
                probabilidade = df_transicoes['contagem'] / somas_por_estado

                # Se a soma for 0 (um estado nunca foi visitado), o resultado da divisão será NaN (Not a Number)
                # Usa-se fillna(0) para tratar esses casos, definindo a probabilidade como 0.
                df_transicoes['probabilidade'] = (probabilidade).fillna(0) 

                # Transforma o formato "longo" (uma transição por linha) para o formato "largo" (matriz),
                # onde o índice são os estados de origem, as colunas são os estados de destino,
                # e os valores são as probabilidades calculadas. 
                matriz_transicao = df_transicoes.pivot(
                    index='estado', 
                    columns='proximo_estado', 
                    values='probabilidade'
                )
                
                # Se algum estado nunca foi ponto de partida, o pivot pode criar NaNs, então garante-se que sejam 0
                matriz_transicao.fillna(0, inplace=True)
                
                # Salvar a matriz
                output_path = transicoes_csv_path.parent / 'matriz_transicao_sentimento.csv'
                matriz_transicao.to_csv(output_path)

            except Exception as e:
                video_path = transicoes_csv_path.parent.parent
                console.print(f'Inválido (salvar_matriz_transicao_sentimento_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição de Sentimento agregada para cada youtuber.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def salvar_matriz_transicao_sentimento_youtuber(youtubers_list: list[str]) -> None:
    # Nomes dos estados de sentimento
    estados_sentimento = ['POS', 'NEU', 'NEG']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matriz de sentimento agregada para [bold cyan]{youtuber}[/bold cyan]')

        try:
            # Encontrar e concatenar todas as transições do youtuber
            lista_dfs_transicoes = []
            for transicoes_csv_path in base_path.rglob('transicoes_sentimento.csv'):
                df_video = pd.read_csv(transicoes_csv_path)
                if not df_video.empty:
                    lista_dfs_transicoes.append(df_video)
            
            if not lista_dfs_transicoes:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições de sentimento encontrado para {youtuber}. Pulando.[/yellow]")
                continue
            
            df_agregado = pd.concat(lista_dfs_transicoes, ignore_index=True)

            # Agrupar por cada tipo de transição e somar as contagens de todos os vídeos
            df_soma_total = df_agregado.groupby(['estado', 'proximo_estado'])['contagem'].sum().reset_index()

            # Garantir que a matriz final seja sempre 3x3, mesmo que algumas transições ou estados nunca tenham ocorrido no agregado de vídeos do youtuber
            tipo_categorico = CategoricalDtype(categories=estados_sentimento, ordered=True)
            df_soma_total['estado'] = df_soma_total['estado'].astype(tipo_categorico)
            df_soma_total['proximo_estado'] = df_soma_total['proximo_estado'].astype(tipo_categorico)
            
            # Calcular a soma das transições que saem de cada estado
            # O .transform('sum') cria uma nova coluna onde cada linha tem a soma total do grupo 'estado' ao qual pertence.
            somas_por_estado = df_soma_total.groupby('estado', observed=False)['contagem'].transform('sum')

            # Calcular a probabilidade de cada transição
            probabilidade = df_soma_total['contagem'] / somas_por_estado

            # Se a soma for 0 (um estado nunca foi visitado), o resultado da divisão será NaN (Not a Number)
            # Usa-se fillna(0) para tratar esses casos, definindo a probabilidade como 0.
            df_soma_total['probabilidade'] = probabilidade.fillna(0)
            
            matriz_transicao_youtuber = df_soma_total.pivot(
                index='estado', 
                columns='proximo_estado', 
                values='probabilidade'
            )
            
            # Garantir que qualquer célula que possa ter ficado como NaN (no caso de um estado nunca ser ponto de partida) seja preenchida com 0.
            matriz_transicao_youtuber.fillna(0, inplace=True)
            
            # Salvar na pasta raiz do youtuber para consistência com outras análises agregadas
            output_path = base_path / 'transicoes' / 'matriz_transicao_sentimento_youtuber.csv'
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição de youtuber gerada: {output_path}")

        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_sentimento_youtuber): {e}')

if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    #salvar_transicoes_sentimento(lista_youtubers)

    #salvar_matriz_transicao_sentimento_video(lista_youtubers)

    salvar_matriz_transicao_sentimento_youtuber(lista_youtubers)
