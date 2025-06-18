import pandas as pd
import re
import os

import spacy
from bertopic import BERTopic

pln = spacy.load("pt_core_news_md")

'''
    Função para percorrer toda a estrutura de pastas dos arquivos coletados
    @param func - Função para ser aplicada dentro da pasta do vídeo
'''
def findVideoFolder(func, columns: list[str]):
    # Definir dados
    base_dir = "files"
    dados = []

    # Percorrer youtubers
    for ytb_folder in os.listdir(base_dir):
        # Criar caminho 'files/ytb_folder'
        next_ytb_dir = os.path.join(base_dir, ytb_folder)

        # Testar se a pasta existe
        if not os.path.isdir(next_ytb_dir):
            # print(f"Erro! [{next_ytb_dir}] não existe! (1)")
            continue

        # Percorrer as pastas dos anos (dentro de 'files/ytb_folder')
        for year_folder in os.listdir(next_ytb_dir):
            # Criar caminho 'files/ytb_folder/year_folder'
            next_year_dir = os.path.join(next_ytb_dir, year_folder)

            # Testar se a pasta existe
            if not os.path.isdir(next_year_dir):
                # print(f"Erro! [{next_year_dir}] não existe! (2)")
                continue

            # Percorrer as pastas dos meses (dentro de 'files/ytb_folder/year_folder')
            for month_folder in os.listdir(next_year_dir):
                # Criar caminho 'files/ytb_folder/year_folder/month_folder'
                next_month_dir = os.path.join(next_year_dir, month_folder)

                # Testar se a pasta existe
                if not os.path.isdir(next_month_dir):
                    # print(f"Erro! [{next_month_dir}] não existe! (3)")
                    continue

                # Percorrer as pastas dos vídeos (dentro de 'files/ytb_folder/year_folder/month_folder/video_folder')
                for video_folder in os.listdir(next_month_dir):
                    # Criar caminho 'files/ytb_folder/year_folder/month_folder/video_folder'
                    next_video_dir = os.path.join(next_month_dir, video_folder)

                    # Testar se a pasta existe
                    if not os.path.isdir(next_video_dir):
                        # print(f"Erro! [{next_video_dir}] não existe! (4)")
                        continue

                    # Chamar função passada como parâmetro para atuar na pasta do vídeo
                    func(next_video_dir, dados, columns)

    # Converter a lista para DataFrame
    df = pd.DataFrame(dados, columns=columns)  # Convertendo a lista para DataFrame

    df2 = df.drop_duplicates(subset=["video_id"])

    # Salvar a lista gerada em um arquivo .csv
    df2.to_csv("kmeans/kmeans_video.csv", index=False)

    # Mostrar mensagem de conclusão
    print(f"Análise concluída com sucesso! ({len(dados)} linhas)")