import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import matplotlib.pyplot as plt

console = Console()
df = pd.read_csv('youtuberslist.csv')
df = df[df['videosTranscritos'] != 0]
YOUTUBER_LIST = df['nome'].tolist()
THRESHOLD = 0.025

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

def calcular_toxicidade_video(sequencias_path):
    sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()

    total_sequencias = len(sequencia) - 1
    total_toxicos = 0
    for estado in sequencia:
        if estado == 'T':
            total_toxicos += 1

    return total_toxicos/total_sequencias
    
def gerar_grafico_toxicidade():
    valores = []
    for youtuber in YOUTUBER_LIST:
        base_path = Path(f'files/{youtuber}')
        for sequencias_path in base_path.rglob("sequencia_toxicidade.csv"):
            toxicidade = calcular_toxicidade_video(sequencias_path)
            valores.append(toxicidade)

    valores = sorted(valores)

    print(valores)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(valores)), valores, marker='.', linestyle='-', markersize=1)
    plt.title("Curva de Toxicidade por Vídeo")
    plt.xlabel("Vídeos (ordenados)")
    plt.ylabel("Toxicidade")
    plt.grid(True)
    plt.show()

def filtro_video(threshold, tirinha_csv_path):
    sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
    return calcular_toxicidade_video(sequencias_path) >= threshold

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
            if filtro_video(THRESHOLD, tirinha_csv_path):
                texto_video_completo = " ".join(tiras_youtuber)
                documento.append(texto_video_completo)
    
    words_count = 0
    for video in documento:
        words_count += len(video.split())

    console.print(f"Dados coletados de {len(YOUTUBER_LIST)} youtubers!")
    console.print(f"-> Quantidade de vídeos do Documento: {len(documento)}")
    console.print(f"-> Quantidade de palavras do Documento: {words_count}\n")

    return documento


# gerar_grafico_toxicidade()