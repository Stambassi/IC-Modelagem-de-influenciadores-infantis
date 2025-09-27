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

if __name__ == "__main__":    
    #lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Geleia', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']

    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    gerar_graficos_youtuber(lista_youtubers)
    #gerar_graficos_tiras(lista_youtubers)

