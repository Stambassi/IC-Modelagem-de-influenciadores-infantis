import os
import json
import pandas as pd

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
                                                

teste = ['Kass e KR','meu nome é david','Lokis']
encontrar_videos_validos(teste)