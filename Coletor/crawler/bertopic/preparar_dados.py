import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path

console = Console()
df = pd.read_csv('youtuberslist.csv')
df = df[df['videosTranscritos'] != 0]
YOUTUBER_LIST = df['nome'].tolist()

def coletar_informacoes_youtuber(video_path) -> str: 
    try:
        with open(video_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = video_data['text']
    except:
        texto_limpo = ""
    return texto_limpo

def coletar_tirinhas_video_janela_tempo(tirinha_csv_path) -> list[str]:
    try:
        sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
        sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()

        tiras_csv = pd.read_csv(tirinha_csv_path)
        tiras = tiras_csv['tiras'].tolist()

        indices_selecionados = set()

        for i, estado in enumerate(sequencia):
            if estado == "T":
                inicio = max(0, i - 3)
                fim = min(len(tiras) - 1, i + 3)
                for j in range(inicio, fim + 1):
                    indices_selecionados.add(j)

        tirinhas_coletadas = [tiras[i] for i in sorted(indices_selecionados)]

        return " ".join(tirinhas_coletadas)

    except Exception as e:
        console.log(e)
        return ""


def coletar_tirinhas_video(tirinha_csv_path, filtro=None) -> list[str]:
    try:
        tiras_csv = pd.read_csv(tirinha_csv_path)
        if filtro == None:
            tiras = tiras_csv['tiras'].tolist()
        else:
            sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
            sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()

            tiras = tiras_csv['tiras'].tolist()

            indices_selecionados = set()

            for i, estado in enumerate(sequencia):
                if estado == filtro:
                    indices_selecionados.add(i-1)

            tiras = [tiras[i] for i in sorted(indices_selecionados)]
            
        return tiras

    except Exception as e:
        console.log(e)
        return []

def get_dados(arquivo_tirinha = 'tiras_video.csv'):
    documento = []
    for youtuber in YOUTUBER_LIST:
        base_path = Path(f'files/{youtuber}')
        for tirinha_csv_path in base_path.rglob(arquivo_tirinha):
            # tiras_youtuber = coletar_tirinhas_video_janela_tempo(tirinha_csv_path)
            # if tiras_youtuber != "":
            #     documento.append(tiras_youtuber)
            #     words_count = len(tiras_youtuber.split())
            #     console.print("Quantidade de palavras: ",words_count)
            tiras_youtuber = coletar_tirinhas_video(tirinha_csv_path)
            texto_video_completo = " ".join(tiras_youtuber)
            documento.append(texto_video_completo)
    
    words_count = 0
    for video in documento:
        words_count += len(video.split())

    console.print(f"Dados coletados de {len(YOUTUBER_LIST)} youtubers!")
    console.print(f"-> Quantidade de vídeos do Documento: {len(documento)}")
    console.print(f"-> Quantidade de palavras do Documento: {words_count}\n")

    return documento


