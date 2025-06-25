import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="pt")

def update_data(sentiment_dict, csv_file):
    try:
        new_row = {
            "negative": sentiment_dict["NEG"],
            "positive": sentiment_dict["POS"],
            "neutral": sentiment_dict["NEU"]
        }
        
        new_df = pd.DataFrame([new_row])
        
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
        
        updated_df.to_csv(csv_file, index=False)
    except Exception as e:
        print(f"Erro: {e}")

def analisar_toxicidade(texto):
    return analyzer.predict(texto).probas

def gerar_graficos(youtuber_list):
    values = ['negative', 'neutral', 'positive']
    for youtuber in youtuber_list:
        df = pd.read_csv(f'./{youtuber}_sentiment.csv')
        for value in values:
            df_value = df[value]

            list_value = df_value.tolist()

            scores = np.array(sorted(list_value))  # Garantir que os dados estejam ordenados

            # Percentis
            percentis = np.linspace(0, 1, len(scores))

            # Aplicando o algoritmo de joelho
            if value == 'neutral':
                knee_locator = KneeLocator(percentis, scores, curve="concave", direction="increasing")
            else:
                knee_locator = KneeLocator(percentis, scores, curve="convex", direction="increasing")

            knee_percentil = knee_locator.knee
            knee_valor = knee_locator.knee_y

            if not os.path.isdir(f'./{youtuber}'):
                os.mkdir(f'./{youtuber}')

            # Plot
            plt.plot(percentis, scores, label='ICDF')
            plt.xlabel("Percentil")
            plt.ylabel(f"Toxicidade ({value})")
            plt.title(f"ICDF da Toxicidade de {youtuber}")
            plt.grid(True)

            # Adiciona o ponto de joelho se existir
            if knee_percentil is not None and knee_valor is not None:
                plt.axvline(knee_percentil, color='r', linestyle='--', label=f'Joelho: {knee_valor:.2f}')
                plt.scatter(knee_percentil, knee_valor, color='red', s=80)
                print(f"Ponto de joelho encontrado: youtuber={youtuber}, value={value}, toxicidade={knee_valor}")
            else:
                print("Nenhum ponto de joelho foi encontrado nos dados.")

            plt.legend()
            plt.savefig(f"./{youtuber}/grafico_{value}.png", dpi=300, bbox_inches='tight')
            plt.close()

def encontrar_videos_validos(youtuber_list):
    """
    Função que encontra todos os vídeos com transcrição e realiza a analise de sentimento
    """
    for youtuber in youtuber_list:
        base_dir = f"../files/{youtuber}"
        csv_file = f"{youtuber}_sentiment.csv"
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
                                    json_path = os.path.join(folder_path, 'video_text.json')
                                    if os.path.exists(json_path): # arquivo tem que existir e ter dados
                                        with open(json_path, 'r') as file:
                                            data = json.load(file)
                                            if data:
                                                update_data(analisar_toxicidade(data['text']),csv_file)          
                                                

lista_youtubers =  ['Kass e KR','Lokis','Julia MineGirl','Luluca Games', 'Papile']
# encontrar_videos_validos(lista_youtubers)
gerar_graficos(lista_youtubers)
