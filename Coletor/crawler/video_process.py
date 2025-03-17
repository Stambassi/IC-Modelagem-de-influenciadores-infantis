import yt_dlp
import os
import wave
import json
import csv
import subprocess
import time

import whisper 
import pandas as pd

from rich.console import Console

console = Console(color_system="auto")
csv_transcripted = "transcripted_videos.csv"
youtuberListPath = "youtuberslist.csv"



def download_youtube_audio(video_id, output_folder):
    """
    Funcao para baixar o video em mp3 do youtube usando a biblioteca 
    yt_dlp (funciona como linha de comando tambem).
    video_id -- id do video para baixar
    output_folder -- pasta para direcionar a saida do download
    return audio -- caminho relativo para o video baixado (com .mp3)
    """

    print(f"> Baixando audio | video_id({video_id})")
    folder = output_folder
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{folder}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True, 
        'no_warnings': True,  
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    console.print("> Download do audio foi um [green]sucesso[/] | video_id("+video_id+")")
    audio = f"{folder}/{video_id}.mp3"
    return audio
    

def transcript_and_delete_audio(audio, model):
    """
    Funcao para transcrever um arquivo de audio utilizando a ferramenta speech-to-text da
    biblioteca Whisper da OpenAI. Alem disso, deleta o audio apos a transcricao
    audio -- caminho do arquivo de audio a ser processado
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    return transcricao -- JSON resposta do whisper {text:"...",segments:[{...}],language:"..."}
    """
    try:
        # Realiza a transcrição
        with console.status("[cyan]Transcrevendo audio...",spinner="dots",refresh_per_second=5.0,speed=0.5):
            modelo = whisper.load_model(model) #, devide = "cpu" // para rodar usando a cpu
            transcricao = modelo.transcribe(audio)
        # Exibe o resultado
        console.print("> Transcrição feita com [green] sucesso [/]")
   
        # Remove o arquivo de áudio após a transcrição
        os.remove(audio)
        console.print(f"> Arquivo deletado com [green] sucesso [/]")
        
        return transcricao
    except Exception as e:
        console.log("[red] Erro [/] ao processar o áudio: ", log_locals=True)
        print(e)
        return None

def result_to_csv(data,output_folder,id):
    """
    Funcao para transformar resultado JSON do speech to text em um arquivo CSV (Nao utilizado)
    data -- JSON a ser convertido
    output_folder -- local onde salvar CSV
    id -- id do video analizado
    """
    print(f"> Criando CSV | path: {output_folder}")
    csv_file = f"{output_folder}/{id}_text_small.csv"
    # Define the CSV column headers
    headers = ['id', 'seek', 'start', 'end', 'text', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob']

    # Write the data to the CSV file
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write the header
        writer.writeheader()
        
        # Write each segment as a row in the CSV
        for segment in data['segments']:
            writer.writerow({
                'id': segment['id'],
                'seek': segment['seek'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'temperature': segment['temperature'],
                'avg_logprob': segment['avg_logprob'],
                'compression_ratio': segment['compression_ratio'],
                'no_speech_prob': segment['no_speech_prob']
            })

        print(f"CSV file '{csv_file}' has been created.")



def video_to_text(video_id, output_folder, model, youtuber):
    """
    Funcao para realizar a transcricao do video e salva-la, assim como criar um csv para armazenar os videos 
    ja analisados
    video_id -- id do video a ser transcrito
    output_folder -- pasta local onde vai ser salvo a transcricao
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes) 
    youtuber -- nome do canal a ser testado
    """
    start_time = time.time()

    local_audio = download_youtube_audio(video_id, output_folder)
    
    transcription_result = transcript_and_delete_audio(local_audio, model)
    json_path = f"{output_folder}/video_text.json"
    with open(json_path, mode='w', encoding='utf-8') as file:
        json.dump(transcription_result, file, ensure_ascii=False, indent=4)
    #result_to_csv(transcription_result,output_folder,video_id)

    if(transcription_result != None):
        data = {'nome': [youtuber], 'video_id': [video_id]}
        df = pd.DataFrame(data)
        try:
            with open(csv_transcripted, 'r'):
                df.to_csv(csv_transcripted, mode='a', header=False, index=False)
        except FileNotFoundError:
            df.to_csv(csv_transcripted, mode='w', header=True, index=False)

    execution_time = time.time() - start_time
    console.print(">>> Tempo de execução do Video_id ("+video_id+") foi de [red]"+str(execution_time)+" segundos [/] [gray]("+str(execution_time/60)+" minutos)[/]")

    return transcription_result

def atualizar_video_transcritos(youtuber, total_videos):
    """
    Funcao para atualizar o total de videos transcritos na planilha youtuberslist.csv dado um numero coletado
    youtuber -- nome do canal do youtube a ser atualizado
    total_videos -- soma de videos a ser atualizado
    """
    df = pd.read_csv(youtuberListPath)
    df.loc[df.nome == youtuber, 'videosTranscritos'] = total_videos
    df.to_csv(youtuberListPath, index=False)

def atualizar_video_total_transcritos(youtuber):
    """
    Funcao para atualizar o total de videos transcritos na planilha youtuberslist.csv de acordo com
    a quantidade de video_text.json que ele tem.     
    youtuber -- nome do canal do youtube a ser atualizado
    return videos -- quantidade de videos transcritos do youtuber
    """
    base_dir = f"files/{youtuber}"
    videos = 0
    if os.path.isdir(base_dir):
        # andar pelos anos
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
                                            videos += 1
                                        # else:
                                        #     print("transcrição vazia")
        atualizar_video_transcritos(youtuber,videos)
    return videos

def process_video(csv, output_folder, model, youtuber):
    """
    Funcao para chamar a transcricao do video evitando a reanalise
    csv -- caminho para o videos_info.csv para coletar o id do youtuber
    output_folder -- pasta local onde vai ser salvo a transcricao
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes) 
    youtuber -- nome do canal a ser testado
    """
    d = pd.read_csv(csv_transcripted)
    df = pd.read_csv(csv)
    video_id = df.loc[0]['video_id']
    video_already_transcripted = False
    found_youtuber = False
    for index, row in d.iterrows():
        if row['nome'] == youtuber:
            found_youtuber = True
            if row['video_id'] == video_id:
                video_already_transcripted = True
                break
        # passou por todos os videos do youtuber e nao achou com id igual
        if found_youtuber and row['nome'] != youtuber:
            break

    if video_already_transcripted:
        console.print("[i]Video ja coletado![/] Passando para o proximo...")
    else:
        video_to_text(video_id, output_folder, model, youtuber)

def process_youtuber_video(model, youtuber):
    """
    Funcao para processar todos os videos de um Youtuber
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    youtuber -- nome do canal que vai ter os videos transcritos
    """
    base_dir = f"files/{youtuber}"
    videos = 0
    youtuber_data = pd.read_csv(youtuberListPath)
    if os.path.isdir(base_dir):
        console.rule("[bold red]Youtuber: "+youtuber)
        # andar pelos anos
        for year_folder in os.listdir(base_dir):
            next_year_dir = os.path.join(base_dir, year_folder)
            if os.path.isdir(next_year_dir):
                # andar pelos meses
                for month_folder in os.listdir(next_year_dir):
                    next_month_dir = os.path.join(next_year_dir, month_folder)
                    if os.path.isdir(next_month_dir):
                        console.log("[bold cyan]"+youtuber+" ("+month_folder+"/"+year_folder+"): ")
                    # andar pelos videos
                        for folder in os.listdir(next_month_dir):
                            folder_path = os.path.join(next_month_dir, folder)
                            if os.path.isdir(folder_path):
                                csv_path = os.path.join(folder_path, 'videos_info.csv')
                                if os.path.exists(csv_path):
                                    console.print("[bold cyan]>>> Transcrevendo Video:[/] "+youtuber+" ("+folder+")", overflow="ellipsis")
                                    process_video(csv_path,folder_path, model, youtuber)
                                    youtuber_data.loc[youtuber_data.nome == youtuber, 'videosTranscritos'] += 1
                                    youtuber_data.to_csv(youtuberListPath)
def process_all_videos(model):
    """
    Funcao para realizar o speech-to-text em todos os videos coletados
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    """
    base_dir = "files"
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        process_youtuber_video(model,ytb_folder)

def gerar_tira(tempo, data_path):
    """
    Funcao para rzealizar o agrupamento dos segments em grupos de X segundos
    tempo -- quanto tempo cada tira vai ter
    data_path -- caminho para arquivo json com resultado da analise do whisper
    """
    margem = 10
    tempo_real = tempo*(1-(margem/100))
    with open(data_path, 'r') as file:
        data = json.load(file)
        total_time = 0
        
        tira_atual = ""
        tiras = []
        for segment in data["segments"]:
            total_time += (segment['end'] - segment['start'])
            tira_atual = tira_atual + segment['text'] 
            if (total_time >= tempo_real):
                tiras.append({"text": tira_atual, "time": total_time})
                tira_atual = ""
                total_time = 0
        if len(tira_atual) > 0:
            tiras.append({"text": tira_atual, "time": total_time})
    
        x = 0
        for i in tiras:
            print(str(x)+": "+i['text']+" ["+str(i["time"])+"s]\n")
            x += 1

        console.print("Total de tiras: "+str(len(tiras)))

def gerar_frases(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)        
        tira_atual = ""
        tiras = []
        for segment in data["segments"]:
            tira_atual = tira_atual + segment['text'] 
            if(tira_atual[len(tira_atual)-1] == '.'):
                tiras.append(tira_atual)
                tira_atual = ""
        if len(tira_atual) > 0:
            tiras.append(tira_atual)
    
        show_tiras(tiras)


        console.print("Total de tiras: "+str(len(tiras)))

def gerar_tira_frase_tempo(tempo, data_path):
    """
    Funcao para realizar o agrupamento dos segments em grupos de X segundos, mantendo a coerencia de frases
    tempo -- quanto tempo cada tira vai ter
    data_path -- caminho para arquivo json com resultado da analise do whisper
    """
    margem = 10
    tempo_real = tempo*(1-(margem/100))
    with open(data_path, 'r') as file:
        data = json.load(file)
        total_time = 0
        
        tira_atual = ""
        tiras = []
        for segment in data["segments"]:
            total_time += (segment['end'] - segment['start'])
            tira_atual = tira_atual + segment['text'] 
            if (total_time >= tempo_real):
                i = tira_atual.rfind(".")
                tiras.append(tira_atual[0:i+1].strip())
                tira_atual = tira_atual[i+1:len(tira_atual)]
                total_time = 0
        if len(tira_atual) > 0:
            tiras.append(tira_atual.strip())
        show_tiras(tiras)
        console.print("Total de tiras: "+str(len(tiras)))
def show_tiras(tiras):
    x = 0
    for i in tiras:
        print(str(x)+": "+i+"\n")
        
        x += 1

def main():
    #process_all_videos("tiny")
   # process_video("/home/stambassi/Documents/Curso/IC/IC-Modelagem-de-influenciadores-infantis/Coletor/crawler/files/Kass e KR/2023/Junho/RESTAURANTE de R$1 vs. RESTAURANTE de R$1.000.000.000 no Minecraft!/videos_info.csv"
    #, "files/Kass e KR/2023/Junho/RESTAURANTE de R$1 vs. RESTAURANTE de R$1.000.000.000 no Minecraft!", "tiny","Kass e KR")
    console.rule("tira por tempo")
    gerar_tira(60,"files/OEPkmsJmY2I_text_small.json")
    console.rule("tira por frase")
    gerar_frases("files/OEPkmsJmY2I_text_small.json")
    console.rule("tira por tempo e frase")
    gerar_tira_frase_tempo(60,"files/OEPkmsJmY2I_text_small.json")

if __name__ == "__main__":
    main()