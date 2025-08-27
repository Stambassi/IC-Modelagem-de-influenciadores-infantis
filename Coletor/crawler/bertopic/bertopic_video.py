import pandas as pd
import re
import os
import json
from rich.console import Console
from bertopic import BERTopic
import spacy

# from rich.console import Console

console = Console()
print("Carregando programa...")
modelagem = BERTopic(language="portuguese")

# Comando para baixar o modelo
# $ python -m spacy download pt_core_news_md
nlp = spacy.load("pt_core_news_md")


def lematizar_json_segment(video_data: json) -> list[str]:
    texto_limpo = []
    for segment in video_data['segments']:
        documento = nlp(segment['text'])
        tokens_segmento = []
        for token in documento:
            if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
                if preservar_tamanho:
                    tokens_segmento.append(str.lower(token.lemma_))
                else:
                    texto_limpo.append(str.lower(token.lemma_))
            elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
                if preservar_tamanho:
                    tokens_segmento.append(str.lower(token.text))
                else:
                    texto_limpo.append(str.lower(token.text))
        if preservar_tamanho: 
            texto_limpo.append(" ".join(tokens_segmento))
    return texto_limpo

def lematizar_json(video_data: json) -> list[str]:
    documento = nlp(video_data['text'])
    texto_limpo = []
    for token in documento:
        if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.lemma_))
        elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.text))
    return texto_limpo

def lematizar_json_max(video_data: json, quantidade_max_token) -> list[str]:
    documento = nlp(video_data['text'])
    quantidade_max_token = len(documento) if quantidade_max_token > len(documento) else quantidade_max_token
    # console.print(f"[red] Quantidade máxima de tokens = {quantidade_max_token}")
    texto_limpo = []
    for token in documento[:quantidade_max_token]:
        if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.lemma_))
        elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.text))
    return texto_limpo

def gerar_topicos_token_variavel(video_path,lista_medias) -> pd.DataFrame(): 
    video_data_path = os.path.join(video_path,"video_text.json")
    topicos = pd.DataFrame()
    try:
        with open(video_data_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            gerou_topico = False
            for i in range(5,1000,5):
                try:
                    if not gerou_topico:
                        texto_limpo = lematizar_json_max(video_data,i)
                        # print(texto_limpo)
                        modelagem.fit_transform(texto_limpo)
                        topicos = modelagem.get_topic_info()
                        console.print(f"Gerou [green]tópicos com {i} tokens")
                        gerou_topico = True
                        # mostrar_topico(topicos)
                        lista_medias.append(i)
                except Exception as error:
                    print(error)
                    gerou_topico = False
    except Exception as e:
        print(e)
    return topicos

def gerar_topicos(video_path, file_path) -> pd.DataFrame(): 
    video_data_path = os.path.join(video_path,"video_text.json")
    topicos = pd.DataFrame()
    try:
        with open(video_data_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = lematizar_json(video_data)
            modelagem.fit_transform(texto_limpo)
            topicos = modelagem.get_topic_info()
            mostrar_topico(topicos)
            gravar_topico(topicos,file_path)

    except Exception as e:
        print(e)
    return topicos

def gravar_topico(topicos,file_path):
    with open(file_path,'a') as f:
        for index, row in topicos.iterrows(): 
            f.write(f"ID: {row['Topic']}\n")
            f.write(f"Nome: {row['Name']}\n")
            f.write(f"Contagem: {row['Count']}\n")
            f.write(f"Conteúdo: {row['Representation']}\n")
            f.write("##########################\n")

def comparar_lematizacao(video_path) -> pd.DataFrame(): 
    video_data_path = os.path.join(video_path,"video_text.json")
    topicos = pd.DataFrame()
    try:
        with open(video_data_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = lematizar_json(video_data)
            console.print(f"Lematização texto completo: [cyan]{len(texto_limpo)} elementos")
            set1 = set(texto_limpo)
            
            texto_limpo = lematizar_json_segment(video_data)
            console.print(f"Lematização texto segmentado: [cyan]{len(texto_limpo)} elementos")
            set2 = set(texto_limpo)

            diff = set1.symmetric_difference(set2)
            print("Different elements:", diff)

    except Exception as e:
        print(e)
    return topicos


def teste_topicos_segmentos(video_path) -> pd.DataFrame(): 
    video_data_path = os.path.join(video_path,"video_text.json")
    topicos = pd.DataFrame()
    try:
        with open(video_data_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = lematizar_json(video_data)
            console.print(f"Lematização texto completo: [cyan]{len(texto_limpo)} elementos")
            
            modelagem.fit_transform(texto_limpo)
            topicos = modelagem.get_topic_info()
            print(topicos.head(10))

            info = modelagem.get_document_info(texto_limpo)
            print(info.columns)
            print(info)

            lemas_segment = lematizar_json_segment(video_data)
            console.print(f"Lematização texto segmentado: [cyan]{len(texto_limpo)} elementos")
            topicos_info = []
            for sublist in info['Top_n_words']:
                # print(sublist)
                for token in sublist.split(" - "):
                    if len(token) >= 0:
                        # print(token)
                        topicos_info.append(token)

            # set_topicos = set([item for sublist in topicos["Representation"] for item in sublist])
            set_topicos = set(
                topicos_info    
            )

            # print(set_topicos)

            print("Elementos dos topicos: ",set_topicos)

            diff = [x for x in lemas_segment if x not in set_topicos]
            print("Elementos fora dos tópicos:", diff)

    except Exception as e:
        print(e)
    return topicos




def mostrar_topico(topicos: pd.DataFrame()):
    for index, row in topicos.iterrows():        
            print("##########################")
            print(f"ID: {row['Topic']}")
            print(f"Contagem: {row['Count']}")
            print(f"Conteúdo: {row['Representation']}")


def lematizar_json_segment_tempo(video_data: json, lista_tempo) -> list[str]:
    texto_limpo = []
    for segment in video_data['segments']:
        documento = nlp(segment['text'])
        tokens_segmento = []
        for token in documento:
            if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
                tokens_segmento.append(str.lower(token.lemma_))
            elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
                    tokens_segmento.append(str.lower(token.text))
        texto_limpo.append(" ".join(tokens_segmento))
    

    # tempo_total = 0
    # novo_tempo = []
    # novo_texto = []
    # tokens_segmento = []
    # print(lista_tempo)
    # for index ,tempo in enumerate(lista_tempo):
    #     tempo_total += tempo
    #     tokens_segmento.append(texto_limpo[index])
    #     if tempo_total >= 60.00:
    #         novo_tempo.append(tempo_total)
    #         novo_texto.append(" ".join(tokens_segmento))
    #         tokens_segmento = []
    #         tempo_total = 0
    # return novo_texto, novo_tempo
    return texto_limpo

def gerar_grafico_topico_tempo(video_path):
    video_data_path = os.path.join(video_path,"video_text.json")
    topicos = pd.DataFrame()
    try:
        with open(video_data_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            tempos = []
            for segment in video_data['segments']:
                # tempos.append(segment['end']-segment['start'])
                tempos.append(segment['end'])

            texto_limpo = lematizar_json_segment_tempo(video_data, tempos)
            console.print("Tamanho texto lematizado: [green]",len(texto_limpo))
            print(texto_limpo)
            console.print("Tamanho dos timestamps: [green]", len(tempos))
            print(tempos)
            modelagem.fit_transform(texto_limpo)
            topicos_tempo = modelagem.topics_over_time(texto_limpo, tempos)

            grafico = modelagem.visualize_topics_over_time(topicos_tempo)
            grafico.write_image("bertopic/Grafico_teste.png")

            print(topicos_tempo)
            print(topicos_tempo['Name'].unique())
            topicos_tempo.to_csv("bertopic/Teste_topicos.csv")
            # print(grafico)
    except Exception as e:
        print(e)
    



'''
    Função para percorrer as pastas de cada vídeo de um youtuber
    @param youtubers_list - Lista de youtubers a serem analisados
    @param function - Função a ser executada na pasta de cada vídeo de um youtuber
''' 
def percorrer_video(youtubers_list: list[str], function) -> None:
    min_topicos = []
    # Percorrer youtubers
    for youtuber in youtubers_list:
        youtuber_file_path = "bertopic"
        youtuber_file_path = os.path.join(youtuber_file_path,f"topicos_{youtuber}.txt")
        base_dir = f"files/{youtuber}"
        if os.path.isdir(base_dir):
            console.rule(youtuber)
            # Percorrer os anos
            for year_folder in os.listdir(base_dir):
                next_year_dir = os.path.join(base_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # Percorrer os meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                            # Percorrer os vídeos
                            for video_folder in os.listdir(next_month_dir):
                                next_video_dir = os.path.join(next_month_dir, video_folder)
                                
                                if os.path.isdir(next_video_dir):
                                    console.print(f"[green]## {video_folder} ##")
                                    with open(youtuber_file_path,'a') as f:
                                        f.write(f"## {video_folder} ##\n")  
                                    function(next_video_dir,youtuber_file_path)
    # avg = sum(min_topicos) / len(min_topicos)
    # console.print(f"Quantidade média de tokens mínimos foi de [green]{avg}")
#lista_youtubers =  ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Jazzghost', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile', 'TazerCraft', 'Tex HS']
lista_youtubers =  ['AuthenticGames']


video_data_path = "files/AuthenticGames/2024/Dezembro/😳 SOBREVIVI em apenas 1 Bloco no Minecraft! (Foi Insano!)"
# topico = gerar_topicos(video_data_path)
# mostrar_topico(topico)
# gerar_grafico_topico(topico)
gerar_grafico_topico_tempo(video_data_path)
# teste_topicos_segmentos(video_data_path)
# percorrer_video(lista_youtubers, comparar_lematizacao)

# percorrer_video(lista_youtubers, gerar_topicos)
