import os
import json
import pandas as pd
from pysentimiento import create_analyzer
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter1d
from rich.console import Console

analyzer = create_analyzer(task="sentiment", lang="pt")
console = Console()
'''
    Função para atualizar os valores de toxicidade dos vídeos de um youtuber
    @param sentiment_dict - Dicionário com os novos valores de toxicidade da análise de sentimento
    @param csv_file_path - Arquivo csv do youtuber a ser atualizado
    @param video_name - Nome do vídeo analisado
'''
def update_data(sentiment_dict: dict, csv_file_path: str, video_name: str) -> None:
    try:
        # Criar novo dicionário no formato do arquivo csv com os valores novos
        new_row = {
            "negative": sentiment_dict["NEG"],
            "positive": sentiment_dict["POS"],
            "neutral": sentiment_dict["NEU"],
            "video_name": video_name
        }
        
        # Crar DataFrame a partir de uma lista [] de dicionários
        new_df = pd.DataFrame([new_row])
        
        # Testar se o caminho do arquivo csv existe
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
        
        # Atualizar o arquivo csv com o DataFrame atualizado
        updated_df.to_csv(csv_file_path, index=False)
    except Exception as e:
        print(f"Erro: {e}")

'''
    Função para analisar a toxicidade de determinado texto
    @param texto - Lista com o trecho a ser analisado pelo modelo
    @return Dict - Dicionário com as probabilidades NEG, NEU e POS
'''
def analisar_toxicidade(texto: str) -> dict:
    return analyzer.predict(texto).probas


'''
    Função para percorrer as pastas dos youtubers, encontrar os vídeos com transcrição e aplicar a análise de sentimento geral
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def atualizar_geral(youtubers_list: list[str]) -> None:
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        csv_file = f"threshold/{youtuber}/sentiment.csv"
        if os.path.isdir(base_dir):
            # andar pelos anos
            print(f">>>>>>>>"+base_dir)
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # andar pelos meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                        # andar pelos videos
                            for folder in os.listdir(next_month_dir):
                                folder_path = os.path.join(next_month_dir, folder)
                                if os.path.isdir(folder_path):
                                    # Analisar o arquivo json com a transcrição
                                    json_path = os.path.join(folder_path, 'video_text.json')
                                    if os.path.exists(json_path): # arquivo tem que existir e ter dados
                                        with open(json_path, 'r') as file:
                                            data = json.load(file)
                                            if data:
                                                if not os.path.isdir(f'threshold/{youtuber}'):
                                                    os.mkdir(f'threshold/{youtuber}')
                                                update_data(analisar_toxicidade(data['text']), csv_file, folder)          


'''
    Função para percorrer as pastas dos youtubers, encontrar os vídeos com transcrição e aplicar a análise de sentimento em cada uma das tiras
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def atualizar_tiras(youtubers_list: list[str]) -> None:
    for youtuber in youtubers_list:
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            # andar pelos anos
            print(f">>>>>>>>"+base_dir)
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # andar pelos meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                        # andar pelos videos
                            for folder in os.listdir(next_month_dir):
                                folder_path = os.path.join(next_month_dir, folder)
                                if os.path.isdir(folder_path):
                                    # Analisar o arquivo csv com as tiras do vídeo
                                    tiras_path = os.path.join(folder_path, 'tiras_video.csv')
                                    if os.path.exists(tiras_path):
                                        data = pd.read_csv(tiras_path)
                                        toxicidade = []
                                        for texto in data['tiras']:
                                            toxicidade.append(analisar_toxicidade(texto)['NEG'])
                                        data['toxicidade'] = toxicidade
                                        data.to_csv(tiras_path, index=False)

'''
    Função para gerar os gráficos ICDF dos valores de cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def gerar_graficos_youtuber(youtubers_list: list[str]) -> None:
    # Criar lista com os nomes das colunas do arquivo csv dos youtubers
    values = ['negative', 'neutral', 'positive']

    # Percorrer a lista de youtubers
    for youtuber in youtubers_list:
        # Testar se a pasta do youtuber existe
        if not os.path.isdir(f'./{youtuber}'):
            print(f'{youtuber} NÃO possui vídeos válidos.')
            continue

        print(f'{youtuber} possui vídeos válidos.')
        # Ler o csv correspondente ao youtuber
        df = pd.read_csv(f'threshold/{youtuber}/sentiment.csv')

        # Percorrer a lista de valores
        for value in values:
            # Separar a coluna do valor específico (negative, neutral, positive)
            df_value = df[value]

            # Converter a coluna em uma lista de floats
            list_value = df_value.tolist()

            # Criar um numpy array a partir da lista de valores ordenados
            scores = np.array(sorted(list_value))  

            # Criar uma lista com n valores igualmente espaçados de 0 a 1 para simular a divisão percentil da lista original
            percentis = np.linspace(0, 1, len(scores))

            # Testar se a pasta do youtuber não existe ainda
            if not os.path.isdir(f'threshold/{youtuber}'):
                os.mkdir(f'threshold/{youtuber}')

            # Calcular o ponto de joelho
            primeira_derivada = np.gradient(scores, percentis)

            #segunda_derivada = gaussian_filter1d(np.gradient(np.gradient(scores, percentis), percentis), sigma=2)
            segunda_derivada = np.gradient(primeira_derivada, percentis)

            try:
                # Calcular o joelho com KneeLocator
                if value == 'neutral':
                    knee = KneeLocator(percentis, scores, curve='convex', direction='increasing')
                else:
                    knee = KneeLocator(percentis, scores, curve='concave', direction='increasing')

                if knee.knee is not None:
                    plt.axvline(knee.knee, color='b', linestyle='--', label=f'Knee: {knee.knee:.2f}')
                    plt.scatter(knee.knee, knee.knee_y, color='blue', s=40)               
                
                # Calcular o joelho com Segunda Derivada
                idx_joelho = np.argmin(segunda_derivada) if value == 'neutral' else np.argmax(segunda_derivada)
                percentil_joelho = percentis[idx_joelho]
                toxicidade_joelho = scores[idx_joelho]

                # Plotar o ponto de joelho
                plt.axvline(percentil_joelho, color='r', linestyle='--', label=f'Derivada: {toxicidade_joelho:.2f}')
                plt.scatter(percentil_joelho, toxicidade_joelho, color='red', s=40)
            except ValueError:
                print(f"'Não existe na lista.")

            # Plotar o gráfico
            plt.plot(percentis, scores, label='ICDF')
            plt.xlabel("Percentil")
            plt.ylabel(f"Toxicidade ({value})")
            plt.title(f"ICDF da Toxicidade de {youtuber}")
            plt.grid(True)

            # Salvar o gráfico
            plt.legend()
            plt.savefig(f"threshold/{youtuber}/grafico_{value}.png", dpi=300, bbox_inches='tight')
            plt.close()

#lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Geleia', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']

lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

# lista_youtubers =  ['AuthenticGames']

# atualizar_geral(lista_youtubers)
atualizar_tiras(lista_youtubers)

#gerar_graficos_youtuber(lista_youtubers)
#gerar_graficos_tiras(lista_youtubers)

