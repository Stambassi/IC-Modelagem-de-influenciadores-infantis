import codecs
import re
import video_process

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta

# Conexão com a API informando o status
from scripts.scriptAPI import connectCheckAPI
from scripts.globalState import GlobalState
from scripts.secondsUntil import secondsUntil
from scripts.console import log

from config import config


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from deep_translator import PonsTranslator

from googletrans import Translator
from transformers import pipeline

import yt_dlp
import os
import wave
import json
import subprocess
import time

from vosk import Model, KaldiRecognizer

import os
import csv
import time
import requests
import json

from rich.console import Console

console = Console()

# Configuração do timeout
import socket
# timeout_in_sec = 15
timeout_in_sec = 60*3 # 3 minutes timeout limit
socket.setdefaulttimeout(timeout_in_sec)

# inicializacao do tradutor para a análise de sentimentos
translator = Translator()

# inicializacaco do modelo BERT para a análise de sentimentos
MODEL  = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


class YouTubeAPIManager:
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    DEVELOPER_KEYS = config["youtube_keys"]
    #DEVELOPER_CHANNELS = config["channel_id"]
    
    static_YouTubeAPIManager = None
    

    def __init__(self):
        self.current_key_index = -1
        #self.current_channel_index = -1
        self.youtube = self.get_new_youtube_client()

    @staticmethod
    def get_instance() -> "YouTubeAPIManager":
        if YouTubeAPIManager.static_YouTubeAPIManager == None:
            YouTubeAPIManager.static_YouTubeAPIManager = YouTubeAPIManager()
        return YouTubeAPIManager.static_YouTubeAPIManager
        
        
    def get_new_youtube_client(self):
        self.DEVELOPER_KEYS = config['youtube_keys']
        #self.DEVELOPER_CHANNELS = config['channel_id']
        if self.current_key_index >= len(self.DEVELOPER_KEYS) - 1:  # Verifica se já tentou todas as chaves
            #timeout = secondsUntil(5)
            timeout = 60
            log("key", f"Todas as chaves excederam a quota. Aguardando {timeout} segundos.")
            GlobalState.get_instance().set_state("status", "sleeping")
            time.sleep(timeout)  # Espera por 24 horas
            GlobalState.get_instance().set_state("status", "working")
            self.current_key_index = 0  # Reinicia o índice da chave para tentar novamente
        else:
            self.current_key_index += 1  # Move para a próxima chave

        #print(f"Id da chave após o incremento/reinicialização: {self.current_key_index}")
        developerKey = self.DEVELOPER_KEYS[self.current_key_index]
        #contadorCanal = contadorCanal + 1
        GlobalState.get_instance().set_state("key_progress", f"{self.current_key_index + 1}/{len(self.DEVELOPER_KEYS)}")
        log("key", f"Usando developerKey: {developerKey}")
        return build(self.YOUTUBE_API_SERVICE_NAME, self.YOUTUBE_API_VERSION, developerKey=developerKey)

    def make_api_request(self, method_func, **kwargs):
        with open("requisições.csv", mode='a', newline='') as csv_file:
            fieldnames = kwargs.keys() 
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            if csv_file.tell() == 0:
                writer.writeheader()
            while True:
                try:
                    writer.writerow(kwargs)
                    # print("salvando")

                    request = method_func(self.youtube, **kwargs)
                    return request.execute()

                
                except HttpError as e:

                    # Define a razão do erro
                    jsonR = e.content if hasattr(e, 'content') else None
                    dados_json = json.loads(jsonR)
                    
                    razao = dados_json["error"]["errors"][0]["reason"]

                    if razao == "quotaExceeded":
                        log("key", "Chave de API atual excedeu a quota, tentando com outra chave...")
                        self.youtube = self.get_new_youtube_client()  # Obtém um novo cliente com a próxima chave
                        log("key", f"Chave atual: {self.current_key_index + 1}/{len(self.DEVELOPER_KEYS)}")
                    elif razao == "commentsDisabled":
                        log("video", f"Video com comentarios desabilitados")
                        raise HttpError(e.resp, e.content, uri=e.uri)
                    elif e.resp.status == 403:
                        print(e)
                        print("Acesso ao vídeo é restrito e requer autorização adicional.")
                        return None
                    else:
                        # log("error", e)
                        print("Erro ao enviar requisição para a API do YouTube - Tentando novamente em ", config["try_again_timeout"], "s")
                        time.sleep(config["try_again_timeout"])

                except Exception as e:
                    try_again_timeout = config["try_again_timeout"]
                    print(f"Problema de conexão com o YouTube detectado - Tentando novamente em {try_again_timeout} segundos.")
                    time.sleep(try_again_timeout)

def create_files_path(nmCanal):
    DEST_DIRECTORY_NAME = f"files/{nmCanal}"

    if not os.path.exists("./" + DEST_DIRECTORY_NAME):
        os.makedirs("./" + DEST_DIRECTORY_NAME)

def create_filesVideo_path(nmCanal, anoVideo, mesVideo, nomeVideo, videoId):

    nomeVideo = limparTitulos(nomeVideo)
    linhaTituloVideo = ""
    resultado = True

    if os.path.exists(f"files/{nmCanal}/videosProcessados.txt"):
        fileRead = open(f"files/{nmCanal}/videosProcessados.txt", "r", encoding="utf-8")
        # console.log("eita",log_locals=True)
        linhaTituloVideo = fileRead.readline()
        while linhaTituloVideo and resultado:
            if(linhaTituloVideo.strip() == videoId.strip()):
                #print(f"Titulo do Video: {nmCanal} ja registrado!!!")
                resultado = False
            linhaTituloVideo = fileRead.readline()

    #if(resultado == True):
        # print(f"Titulo do Video: {nomeVideo}")
        #resultado = False
        #a = input()

    if (resultado == True):
        fileWrite = open(f"files/{nmCanal}/videosProcessados.txt", "a", encoding="utf-8") 
        fileWrite.write(f"{videoId}\n")
        fileWrite.close()

    DEST_DIRECTORY_NAME = f"files/{nmCanal}/{anoVideo}/{mesVideo}/{nomeVideo}"

    if not os.path.exists("./" + DEST_DIRECTORY_NAME):
        os.makedirs("./" + DEST_DIRECTORY_NAME)

    return resultado

def limparTitulos (nmVideo):

    """
    Remove caracteres inválidos para nomes de arquivo/diretório.
    """

    i = 0
    caracter = ''
    while(i < len(nmVideo)):
        
        caracter = nmVideo[i]

        match caracter:
            case '<':
                nmVideo = nmVideo.replace('<', '_')
            case '>': 
                nmVideo = nmVideo.replace('>', '_')
            case ':': 
                nmVideo = nmVideo.replace(':', '_')
            case '"': 
                nmVideo = nmVideo.replace('"', '_')
            case '/': 
                nmVideo = nmVideo.replace('/', '_')
            case '|': 
                nmVideo = nmVideo.replace('|', '_')
            case '?': 
                nmVideo = nmVideo.replace('?', '_')
            case '*': 
                nmVideo = nmVideo.replace('*', '_')
            case 'ã':
                nmVideo = nmVideo.replace('ã','a')
            case 'Ã':
                nmVideo = nmVideo.replace('Ã','A')
        i += 1

    #print(f"O nome do video eh: {nmVideo} \n")
    return nmVideo
            

def generate_date_intervals(start_date, end_date, interval_type):
    interval_delta = {"weekly": timedelta(weeks=1), "monthly": timedelta(days=30)}
    current_start = end_date
    while current_start > start_date:
        current_end = min(current_start, end_date)
        current_start = max(current_end - interval_delta[interval_type], start_date)
        yield current_start, current_end

def is_short_video(video_id):
    shorts_url = f"https://www.youtube.com/shorts/{video_id}"
    response = requests.head(shorts_url, allow_redirects=False)
    return response.status_code == 200

def get_video_details(video_id):
    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    method_func = lambda client, **kwargs: client.videos().list(**kwargs)
    print(">> Request video detais")
    video_response = api_manager.make_api_request(method_func, id=video_id, 
                                                  part='snippet,statistics,contentDetails,status,liveStreamingDetails,localizations,topicDetails,recordingDetails')

    if video_response == None:
        return None
    video_details = video_response['items'][0]
    snippet = video_details['snippet']
    contentDetails = video_details['contentDetails']
    status = video_details['status']
    statistics = video_details['statistics']
    processingDetails = video_details.get('processingDetails', {})
    liveStreamingDetails = video_details.get('liveStreamingDetails', {})
    localizations = video_details.get('localizations', {})
    topicDetails = video_details.get('topicDetails', {})
    recordingDetails = video_details.get('recordingDetails', {})

    # Constrói o dicionário com os detalhes do vídeo
    details = {
        "video_id": video_id,
        "title": snippet.get('title'),
        "description": snippet.get('description'),
        "channel_id": snippet.get('channelId'),
        "published_at": snippet.get('publishedAt'),
        "category_id": snippet.get('categoryId', ""),
        "tags": snippet.get('tags', []),
        "view_count": int(statistics.get('viewCount', 0)),
        "like_count": int(statistics.get('likeCount', 0)),
        "comment_count": int(statistics.get('commentCount', 0)),
        "duration": contentDetails.get('duration'),
        "definition": contentDetails.get('definition'),
        "caption": contentDetails.get('caption') == 'true',
        "licensed_content": contentDetails.get('licensedContent', False),
        "privacy_status": status.get('privacyStatus'),
        "license": status.get('license'),
        #"embeddable": status.get('embeddable', False),
        "public_stats_viewable": status.get('publicStatsViewable', False),
        "is_made_for_kids": status.get('madeForKids', False),
        #"thumbnail_url": snippet.get('thumbnails', {}).get('high', {}).get('url'),
        "default_audio_language": snippet.get('defaultAudioLanguage'),
        "default_language": snippet.get('defaultLanguage'),
        "actual_start_time": liveStreamingDetails.get('actualStartTime', ''),
        "scheduled_start_time": liveStreamingDetails.get('scheduledStartTime', ''),
        "actual_end_time": liveStreamingDetails.get('actualEndTime', ''),
        "scheduled_end_time": liveStreamingDetails.get('scheduledEndTime', ''),
        "concurrent_viewers": liveStreamingDetails.get('concurrentViewers', 0),
        "active_live_chat_id": liveStreamingDetails.get('activeLiveChatId', ''),
        "recording_date": recordingDetails.get('recordingDate', ''),
        "topicCategories": topicDetails.get('topicCategories', []),
        "processing_status": processingDetails.get('processingStatus', ''),
        "parts_total": processingDetails.get('processingProgress', {}).get('partsTotal', 0),
        "parts_processed": processingDetails.get('processingProgress', {}).get('partsProcessed', 0),
        "time_left_ms": processingDetails.get('processingProgress', {}).get('timeLeftMs', 0),
        "processing_failure_reason": processingDetails.get('processingFailureReason', '')}
    
    res = not any(details.values()) 
    if res == True:
        print("Valores do get video detais")             

    return details

def get_comments(video_id, total_comment_count):
    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto

    comments_data = []
    page_token = None
    collected_comments = 0
    while True:
        try:
            method_func = lambda client, **kwargs: api_manager.youtube.commentThreads().list(**kwargs)
            # print(">> Request de comentarios")
            response = api_manager.make_api_request(method_func,part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText")
            
            # print(response)
        except HttpError as e:
            jsonR = e.content if hasattr(e, 'content') else None
            dados_json = json.loads(jsonR)
            
            razao = dados_json["error"]["errors"][0]["reason"]
            if e.resp.status == 404:
                log("video", f"Vídeo com ID {video_id} não encontrado. Pulando...")
                return comments_data  # Retorna os dados de comentários coletados até agora
            elif razao == "commentsDisabled":
                log("video", f"Vídeo com ID {video_id} possui comentários bloqueados. Pulando...")
                return comments_data  # Retorna os dados de comentários coletados até agora
            else:
                print(e["detailes"])
                raise e  # Releva outros erros para serem tratados externamente
    

        for item in response.get("items", []):
            collected_comments +=1  # Inicializa o contador de comentários coletado
            comment_info = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["snippet"]["topLevelComment"]["id"]

            #Pegar o conteudo principal do comentario 
            ##comment_content = comment_info.get("textDisplay")
            #Traduzir ele para inglês
            ##comment_content = traducaoPTEN(comment_content)
            #Rodar analise de sentimentos BERT
            ##resultadoSentimentos = sentiment_analisys(comment_content)

            comments_data.append({
                "video_id": video_id,
                "comment_id": comment_id,
                "author": comment_info.get("authorDisplayName"),
                #"author_profile_image_url": comment_info.get("authorProfileImageUrl"),
                "author_channel_url": comment_info.get("authorChannelUrl"),
                "author_channel_id": comment_info.get("authorChannelId", {}).get("value"),
                "comment": comment_info.get("textDisplay"),
                "published_at": comment_info.get("publishedAt"),
                "updated_at": comment_info.get("updatedAt", None),  # Pode não estar presente em todas as respostas
                "like_count": comment_info.get("likeCount"),
                "viewer_rating": comment_info.get("viewerRating", ""), 
                "can_rate": comment_info.get("canRate", ""),
                "is_reply": False,
                "parent_id": None,
                #"roberta-neg": resultadoSentimentos[0],
                #"roberta-neu": resultadoSentimentos[1],
                #"roberta-pos": resultadoSentimentos[2]
            })
            # Verifique se o comentário tem respostas e as colete
            total_reply_count = item["snippet"]["totalReplyCount"]            
            if total_reply_count > 0:
                # print(">> Coletando replies")
                replies = get_replies(video_id, comment_id)
                comments_data.extend(replies)
        page_token = response.get('nextPageToken')
        if not page_token:
            break
    res = not any(comments_data) 
    if res == True:
        log("comments", "Valores do get comments vazios")             

    console.print(f"Coletados [green]{collected_comments}[/] de {total_comment_count} comentários para o vídeo {video_id}.")
    return comments_data

def get_replies(video_id, comment_id):
    replies_data = []
    page_token = None
    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    #print(">> Request de replies")
    while True:
        method_func = lambda client, **kwargs: api_manager.youtube.comments().list(**kwargs)
        nextPageToken = None
        try:

            response = api_manager.make_api_request(method_func,part="snippet",
                parentId=comment_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText")

            if(nextPageToken == response.get('nextPageToken')):
                nextPageToken = None
            else:
                nextPageToken = response.get('nextPageToken')

            
            # print("Salvando ", len(response.get("items", [])), " replies únicos", video_id)

            for item in response.get("items", []):
                reply_info = item["snippet"]
                replies_data.append({
                    "video_id": video_id,
                    "comment_id": item["id"],
                    "author": reply_info.get("authorDisplayName"),
                    "author_profile_image_url": reply_info.get("authorProfileImageUrl"),
                    "author_channel_url": reply_info.get("authorChannelUrl"),
                    "author_channel_id": reply_info.get("authorChannelId", {}).get("value"),
                    "comment": reply_info.get("textDisplay"),
                    "published_at": reply_info.get("publishedAt"),
                    "updated_at": reply_info.get("updatedAt", ""), 
                    "like_count": reply_info.get("likeCount"),
                    "viewer_rating": reply_info.get("viewerRating", ""), 
                    "can_rate": reply_info.get("canRate", ""),
                    "is_reply": True,
                    "parent_id": comment_id        
                })

            # page_token = response.get('nextPageToken')
            # print("reply page token: ", nextPageToken)
            if not page_token:
                break
        except HttpError as e:
            print("Erro ocorreu ao coletar replies")
            with open(f"consulta_{video_id}_{comment_id}.txt", "w") as file: 
                file.write(e) 
            break
    
    res = not any(replies_data) 
    return replies_data

def get_channel_details(channel_id):
    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    method_func = lambda client, **kwargs: api_manager.youtube.channels().list(**kwargs)
    # print(">> Request de detalhes do canal")
    channel_response = api_manager.make_api_request(method_func,
        part="snippet,statistics,contentDetails,brandingSettings",
        id=channel_id)

    if not channel_response.get('items'):
        return None  # Retorna None se não encontrar detalhes do canal

    channel_details = channel_response['items'][0]
    snippet = channel_details['snippet']
    statistics = channel_details['statistics']
    brandingSettings = channel_details.get('brandingSettings', {})

    # Coleta de informações básicas e adicionais do canal
    details = {
        "channel_id": channel_id,
        "title": snippet.get('title', ""),
        "description": snippet.get('description', ""),
        "published_at": snippet.get('publishedAt', ""),
        "country": snippet.get('country', ""),  
        "view_count": int(statistics.get('viewCount', 0)),
        "comment_count": int(statistics.get('commentCount', 0)),
        "subscriber_count": int(statistics.get('subscriberCount', 0)),
        "video_count": int(statistics.get('videoCount', 0)),
        "is_verified": brandingSettings.get('channel', {}).get('isVerified', False),
        "keywords": brandingSettings.get('channel', {}).get('keywords', ""),  # Palavras-chave do canal
        "profile_picture_url": snippet.get('thumbnails', {}).get('default', {}).get('url', ""),  # URL da imagem de perfil
    }

    res = not any(details.values()) 
    if res == True:
        log("channels", "Valores do get channel details vazios") 
    
    return details

# Funcao para atualizar a data de coleta na lista de youtubers
def atualizarUltimaDatadeColeta(nmCanal, mesPublicacaoVideo, anoPublicacaoVideo):
    youtuberListPath = "youtuberslist.csv"
    channel_data  = pd.read_csv(youtuberListPath)
    for i in range(len(channel_data.index)):
        nomeAtual = channel_data['nome'].loc[channel_data.index[i]]
        if(nomeAtual == nmCanal):
            ultimoAno = int(channel_data['ultimoAnoColetado'].loc[channel_data.index[i]])
            ultimoMes = channel_data['ultimoMesColetado'].loc[channel_data.index[i]]
            anoPublicacaoVideo = int(anoPublicacaoVideo)
            # console.log("teste",log_locals = True)
            if(anoPublicacaoVideo > ultimoAno):
                channel_data.at[channel_data.index[i], 'ultimoAnoColetado'] = anoPublicacaoVideo
                channel_data.at[channel_data.index[i], 'ultimoMesColetado'] = mesPublicacaoVideo
                channel_data.to_csv(youtuberListPath, index=False)
                console.print("[cyan] Data atualizada!! [/]")
            elif(anoPublicacaoVideo == ultimoAno and numeroMesAno(mesPublicacaoVideo) > numeroMesAno(ultimoMes)):
                channel_data.at[channel_data.index[i], 'ultimoMesColetado'] = mesPublicacaoVideo
                channel_data.to_csv(youtuberListPath, index=False)
                console.print("[cyan] Data atualizada!! [/]")

            
# Função para processar um único vídeo
def process_video(video_id, processed_videos, nmCanal, video_details, anoPublicacaoVideo, mesPublicacaoVideo):
    global channels_info
    tituloVideo = video_details['title']
    resposta = create_filesVideo_path(nmCanal, anoPublicacaoVideo, mesPublicacaoVideo, tituloVideo, video_id)
    #print(f"Conseguiu criar arquivo para o video: {resposta}")
    if(resposta == True):
        #console.log("antes de limpar",log_locals=True)
        tituloVideo = limparTitulos(tituloVideo)
        # console.log("depois de limpar",log_locals=True)
        print(">> processando vídeos")
        videos_file_exists = os.path.isfile(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/videos_info.csv')
        #channels_file_exists = os.path.isfile(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/channels_info.csv')
        comments_file_exists = os.path.isfile(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/comments_info.csv')

        #video_details = get_video_details(video_id)
        if video_details == None:
            print("Erro por causa de autorização")
            return
        total_comment_count = video_details['comment_count']  # Assumindo que 'comment_count' é o total de comentários disponíveis

        if total_comment_count > 0 and total_comment_count < 10000000000000: #Sentinel 
            pd.DataFrame([video_details]).to_csv(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/videos_info.csv', mode='a', header=not videos_file_exists, index=False)
            
            # channel_details = get_channel_details(video_details['channel_id'])
            # pd.DataFrame([channel_details]).to_csv(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/channels_info.csv', mode='a', header=not channels_file_exists, index=False)
            
            comments = get_comments(video_id, total_comment_count)
            comments_df = pd.DataFrame(comments)
            comments_df['channel_id'] = video_details['channel_id']
            
            comments_df.to_csv(f'files/{nmCanal}/{anoPublicacaoVideo}/{mesPublicacaoVideo}/{tituloVideo}/comments_info.csv', mode='a', header=not comments_file_exists, index=False, quoting=csv.QUOTE_MINIMAL)

def make_search_request(query, published_after, published_before, REGION_CODE, RELEVANCE_LANGUAGE, channel_id):    
    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    method_func = lambda client, **kwargs: api_manager.youtube.search().list(**kwargs)
    #print("CHAMANDO")
    # print("Precisa adicionar o parâmetro location e radius para configurarmos os países")
   # https://developers.google.com/youtube/v3/docs/search/list (Adicionar o parâmetro location e radius...)
    console.print(f">>> Nova query [green]({query})[/]")
    search_response = api_manager.make_api_request(method_func,
    part="id,snippet",
    q=query,
    maxResults=50,
    type="video",
    order="relevance",
    publishedAfter=published_after,
    publishedBefore=published_before,
    regionCode=REGION_CODE,
    relevanceLanguage=RELEVANCE_LANGUAGE,
    channelId=channel_id )
    number_of_videos = len(search_response.get('items', []))
    print(f"A requisição da query retornou {number_of_videos} vídeos.")


    return search_response

def nomeCanal(channel_id):

    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    method_func = lambda client, **kwargs: api_manager.youtube.channels().list(**kwargs)
    print("Nova chamada de nome")
    search_response = api_manager.make_api_request(method_func,
    part="id,snippet",
    id = channel_id
    )
    

    if 'items' in search_response and len(search_response['items']) > 0:
            channel_info = search_response['items'][0]['snippet']
            channel_name = channel_info['title']
            return channel_name
    else:
        print("Nenhum canal encontrado com esse ID")
        return None
    
def nomeMesAno(numeroMes):

    stringMes = ""

    match numeroMes:
        case "01":
            stringMes = "Janeiro"
        case "02":
            stringMes = "Fevereiro"
        case "03":
            stringMes = "Marco"
        case "04":
            stringMes = "Abril"
        case "05":
            stringMes = "Maio"
        case "06":
            stringMes = "Junho"
        case "07":
            stringMes = "Julho"
        case "08":
            stringMes = "Agosto"
        case "09":
            stringMes = "Setembro"
        case "10":
            stringMes = "Outubro"
        case "11":
            stringMes = "Novembro"
        case "12":
            stringMes = "Dezembro"

    return stringMes

def numeroMesAno(stringMes):

    numeroMes = 0

    match stringMes:
        case "Janeiro":
            numeroMes = 1
        case "Fevereiro":
            numeroMes = 2
        case "Marco":
            numeroMes = 3
        case "Abril":
            numeroMes = 4
        case "Maio":
            numeroMes = 5
        case "Junho":
            numeroMes = 6
        case "Julho":
            numeroMes = 7
        case "Agosto":
            numeroMes = 8
        case "Setembro":
            numeroMes = 9
        case "Outubro":
            numeroMes = 10
        case "Novembro":
            numeroMes = 11
        case "Dezembro":
            numeroMes = 12

    return numeroMes

def traducaoPTEN(text):
    try: 
        translated_text = translator.translate(text, src='pt', dest='en').text
        return translated_text
    except:
        return text

def sentiment_analisys(text): 

    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta-neg': scores[0],
        'roberta-neu': scores[1],
        'roberta-pos': scores[2],
    }
    #print(scores_dict)
    return scores
    
def main():

   # Configurar com aspas duplas os termos chaves -> testar primeiro....

    youtuberListPath = "youtuberslist.csv"
    channel_data  = pd.read_csv(youtuberListPath)
    queries = config["queries"]
    youtubers = channel_data['nome']

    df_atual_date = pd.read_csv('files/atual_date.csv', header=None)

    GlobalState.get_instance().set_state("status", "working")

    # Captura a data atual de busca do dataframe atual_date (year, month, day)
    atual_date = {
        "year": df_atual_date.iloc[0, 0],
        "month": df_atual_date.iloc[0, 1],
        "day": df_atual_date.iloc[0, 2],
    }

    start_date = datetime(config['start_date'][0], config['start_date'][1], config['start_date'][2]) #Data inicial da coleta
    #end_date = datetime(config['end_date'][0], config['end_date'][1], config['end_date'][2])
    end_date = datetime(atual_date["year"], atual_date["month"], atual_date["day"]) #Data final 
    interval_type = "monthly" #Intervalo da busca, se é mensal(monthtly), ou semanal (weekly)
    REGION_CODE = config['region_code']
    RELEVANCE_LANGUAGE = config['relevance_language']
    TOP_COMMENTED = False #Pegar os vídeos mais comentados? Não vale a pena porque está retornando vídeo do tema...
    number_of_videos_to_process = 0
    REQUIRE_TITLE_KEYWORDS = False # Forçar o processamento dos vídeos e comentários com determinadas keywords nos títulos...
    
    processed_videos = set()

    # Tenta carregar vídeos já processados
    try:
        with open('files/processed_videos.csv', 'r') as file:
            processed_videos = {row[0] for row in csv.reader(file)}
    except FileNotFoundError:
        pass  # Continua com o conjunto vazio se o arquivo não existir

    api_manager = YouTubeAPIManager.get_instance()  # Obtendo a instância do objeto
    
    connectCheckAPI() # Conecta com a API de status -> Caso não configurou, ignore


    # Comeco da coleta de dados, implementação funciona com 3 repeticoes: A maior sobre intervalo de datas, determinado pelo arquivo atual_date
    # e as datas no config, depois por cada youtuber e por ultimo pelas querys

    for start_interval, end_interval in generate_date_intervals(start_date, end_date, interval_type):
        # Atualiza o atual_date.csv para cada iteração do gerador de intervalos
        with open("files/atual_date.csv", "w", newline="") as csvfile:
            fieldnames = ["year", "month", "day"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({
                "year": end_interval.year,
                "month": end_interval.month,
                "day": end_interval.day
            })

        # Aqui ocorre o loop para a coleta de dados (Realiza uma vez para cada query com um youtuber depois muda o youtuber)     
        for youtuber in youtubers:

            console.rule(f"Youtuber: {youtuber} ({start_interval} - {end_interval})")
            channel_id = channel_data.loc[channel_data['nome'] == youtuber, 'channel_id'].item()
            create_files_path(youtuber) # Cria diretório files para armazenar saidas

            for query in queries:
    
                print(f">>> Query: {query}")
                GlobalState.get_instance().set_state("atual_query", query)
                #GlobalState.get_instance().set_state("query_progress", f"{queries.index(query) + 1}/{quantidadeQuerys}")
            
                published_after = start_interval.isoformat() + "Z"
                published_before = end_interval.isoformat() + "Z"
                video_details_list = []
                #print(channel_data)
                search_response = make_search_request(query, published_after, published_before, REGION_CODE, RELEVANCE_LANGUAGE,channel_id) 
                videos = search_response.get("items", [])
                total_videos = len(videos)
                    
                if total_videos == 0:  # Verifica se search_response foi obtido com sucesso
                    console.log("[red]Não foi possível obter uma resposta da API.[/] Movendo para a próxima consulta.")
                    continue
                
                for index, item in enumerate(videos, start=1):
                    
                    VIDEO_TITLE = item['snippet']['title'].lower()

                    key_words = config['key_words']

                    # Verifica se o título possui as palavras chave
                    if any((word.lower() in VIDEO_TITLE for word in key_words) or len(key_words) == 0):
                        video_id = item['id']['videoId']
                        print(f"Processando vídeo {index} de {total_videos}: ID = {video_id}")
                        if video_id not in processed_videos:
                            video_details = get_video_details(video_id)
                            comment_count = video_details['comment_count']
                            data_publicacao_Video = video_details['published_at']
                            
                            anoPublicacaoVideo = data_publicacao_Video[0:4]
                            mesPublicacaoVideo = nomeMesAno(data_publicacao_Video[5:7])

                            
                            #a = input('').split("")[0]
                            #print(a)
                        
                            console.print(f"[cyan]Título[/]: {video_details['title']}, Quantidade de comentários: [bold green]{video_details['comment_count']}[/]")
                            atualizarUltimaDatadeColeta(youtuber,mesPublicacaoVideo,anoPublicacaoVideo)
                            if comment_count > 0:
                                process_video(video_id, processed_videos, youtuber, video_details, anoPublicacaoVideo, mesPublicacaoVideo)

                console.log(f"Coleta concluída para a consulta: {query} entre {start_interval} e {end_interval}")
                console.print(">> Canal analisado foi: [bold green]"+youtuber+"[/]")
                

if __name__ == "__main__":
    main()
