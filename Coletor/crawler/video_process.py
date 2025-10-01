import yt_dlp
import os
import json
import csv
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
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_folder}/%(id)s.%(ext)s',
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
    audio = f"{output_folder}/{video_id}.mp3"
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

def atualizar_csv_videos_processados(youtuber, video_id):
    nova_linha = {'nome': [youtuber], 'video_id': [video_id]}
    linha_list = [youtuber,video_id]
    linha_igual = False
    try:
        df = pd.read_csv(csv_transcripted)
        if df.empty:
            df = pd.DataFrame(nova_linha)
            df.to_csv(csv_transcripted, mode='w', header=True, index=False)
        else:
            for _, row in df.iterrows():
                vid_id = row['video_id']
                #print(vid_id +" == "+video_id)
                if vid_id == video_id:
                    linha_igual = True
            if not linha_igual:
                #print(linha_list)
                df.loc[len(df)] = linha_list
                #print("Row inserted successfully.")
                with open(csv_transcripted, 'r'):
                    df.to_csv(csv_transcripted, mode='w', header=True, index=False)
    except FileNotFoundError:
        df = pd.DataFrame(nova_linha)
        df.to_csv(csv_transcripted, mode='w', header=True, index=False)
    

def video_to_text(video_id, output_folder, model, youtuber, local_audio = None):
    """
    Funcao para realizar a transcricao do video e salva-la, assim como criar um csv para armazenar os videos 
    ja analisados
    video_id -- id do video a ser transcrito
    output_folder -- pasta local onde vai ser salvo a transcricao
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes) 
    youtuber -- nome do canal a ser testado
    """
    start_time = time.time()

    if local_audio == None:
        local_audio = download_youtube_audio(video_id, output_folder)
    
    transcription_result = transcript_and_delete_audio(local_audio, model)
    json_path = f"{output_folder}/video_text.json"
    with open(json_path, mode='w', encoding='utf-8') as file:
        json.dump(transcription_result, file, ensure_ascii=False, indent=4)
        console.print("-- Caminho: "+json_path)
    #result_to_csv(transcription_result,output_folder,video_id)
    if transcription_result != None:
        atualizar_csv_videos_processados(youtuber,video_id)
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
    df = pd.read_csv(youtuberListPath)

    if os.path.isdir(base_dir) and youtuber in df.values:
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
                                            data_path = os.path.join(folder_path, 'videos_info.csv')
                                            try:
                                                dados_video = pd.read_csv(data_path)
                                                vid_id = dados_video.at[0,'video_id']
                                                atualizar_csv_videos_processados(youtuber,vid_id)
                                            except FileNotFoundError:
                                                console.log("Error",log_locals=True)
                                        # else:
                                        #     print("transcrição vazia")
        atualizar_video_transcritos(youtuber,videos)
    return videos

def video_already_transcripted(youtuber, video_id):
    d = pd.read_csv(csv_transcripted)
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
    return video_already_transcripted

def process_video(csv, output_folder, model, youtuber, audio = None):
    """
    Funcao para chamar a transcricao do video evitando a reanalise
    csv -- caminho para o videos_info.csv para coletar o id do youtuber
    output_folder -- pasta local onde vai ser salvo a transcricao
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes) 
    youtuber -- nome do canal a ser testado
    """
    df = pd.read_csv(csv)
    video_id = df.loc[0]['video_id']

    if video_already_transcripted(youtuber,video_id):
        console.print("[i]Video ja coletado![/] Passando para o proximo...")
    else:
        if audio == None: 
            video_to_text(video_id, output_folder, model, youtuber)
        else:
            audio_path = os.path.join(output_folder,audio)
            if os.path.isfile(audio_path):
                video_to_text(video_id, output_folder, model, youtuber,local_audio = audio_path)
            else:
                print(audio)
                console.print("[red]Vídeo sem áudio![/] Passando para o proximo...")


def process_youtuber_video(model, youtuber):
    """
    Funcao para processar todos os videos de um Youtuber
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    youtuber -- nome do canal que vai ter os videos transcritos
    """
    start()
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
                                    try:
                                        console.print("[bold cyan]>>> Transcrevendo Video:[/] "+youtuber+" ("+folder+")", overflow="ellipsis")
                                        process_video(csv_path,folder_path, model, youtuber)
                                        youtuber_data.loc[youtuber_data.nome == youtuber, 'videosTranscritos'] += 1
                                        youtuber_data.to_csv(youtuberListPath, index=False)
                                    except Exception as e:
                                        console.log(f"[red]ERRO[/]: {e}",log_locals=True)
def process_all_videos(model):
    """
    Funcao para realizar o speech-to-text em todos os videos coletados
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    """
    base_dir = "files"
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        process_youtuber_video(model, ytb_folder)

def gerar_tira(tempo, data_path):
    """
    Funcao para realizar o agrupamento dos segments em grupos de X segundos
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

def start():
    df = pd.read_csv(csv_transcripted)
    df.sort_values('nome',ascending=True, inplace=True)
    df.to_csv(csv_transcripted, mode='w', header=True, index=False)

def gerar_tira_frase_tempo(tempo, data_path):
    """
    Funcao para realizar o agrupamento dos segments em grupos de X segundos, mantendo a coerencia de frases
    tempo -- quanto tempo cada tira vai ter
    data_path -- caminho para arquivo json com resultado da analise do whisper
    """
    margem = 10
    tempo_real = tempo * (1 - (margem / 100))
    with open(data_path, 'r') as file:
        data = json.load(file)
        total_time = 0
        
        tira_atual = ""
        tiras = []
        for segment in data["segments"]:
            total_time += (segment['end'] - segment['start'])
            tira_atual = tira_atual + segment['text'] 
            if (total_time >= tempo_real):
                i = max(tira_atual.rfind("."), tira_atual.rfind("?"), tira_atual.rfind("!"))
                if i < 0: 
                    i = len(tira_atual)
                tiras.append(tira_atual[0:i+1].strip())
                tira_atual = tira_atual[i+1:len(tira_atual)]
                total_time = 0
        if len(tira_atual) > 0:
            tiras.append(tira_atual.strip())
        # show_tiras(tiras)
        # console.print("Total de tiras: "+str(len(tiras)))

        return tiras

def show_tiras(tiras):
    x = 0
    for i in tiras:
        print(str(x)+": "+i+"\n")
        
        x += 1

'''
    Função para percorrer toda a estrutura de pastas dos arquivos coletados e armazenas as tiras em um arquivo csv
'''
def save_tiras():
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

                    # Definir caminho para o arquivo .json com as tiras de cada vídeo
                    csv_path = os.path.join(next_video_dir, 'video_text.json')

                    # Testar se o arquivo existe
                    if not os.path.exists(csv_path):
                        continue

                    # Calcular as tiras de cada vídeo
                    tiras = gerar_tira_frase_tempo(60, csv_path)

                    # Converter a lista para DataFrame
                    df_tiras = pd.DataFrame(tiras, columns=['tiras'])  

                    # Salvar a lista gerada em um arquivo .csv
                    df_tiras.to_csv(f"{next_video_dir}/tiras_video.csv", index_label='index')

def download_videos_youtuber(youtuber):
    """
    Funcao para baixar todos os videos de um Youtuber
    youtuber -- nome do canal que vai ter os videos transcritos
    """
    start()
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
                                    try:
                                        df = pd.read_csv(csv_path)
                                        video_id = df.iloc[0]['video_id']

                                        if not video_already_transcripted(youtuber,video_id):
                                            console.print("[bold cyan]>>> Baixando Video:[/] "+youtuber+" ("+folder+")", overflow="ellipsis")
                                            download_youtube_audio(video_id, folder_path)
                                        else:
                                            console.print("[bold green]>>> Vídeo já transcrito -- pulando download [/] "+youtuber+" ("+folder+")", overflow="ellipsis")

                                    except Exception as e:
                                        console.log(f"[red]ERRO[/]: {e}",log_locals=True)

def download_all_videos():
    """
    Funcao para realizar o speech-to-text em todos os videos coletados
    """
    base_dir = "files"
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        download_videos_youtuber(ytb_folder)

def transcript_videos_youtuber(model, youtuber):
    """
    Funcao para processar todos os videos de um Youtuber a partir de audios baixados
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    youtuber -- nome do canal que vai ter os videos transcritos
    """
    start()
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
                                    try:
                                        console.print("[bold cyan]>>> Transcrevendo Video:[/] "+youtuber+" ("+folder+")", overflow="ellipsis")
                                        df = pd.read_csv(csv_path)
                                        video_id = df.iloc[0]['video_id']
                                        audio = f"{video_id}.mp3"
                                        process_video(csv_path,folder_path, model, youtuber, audio = audio)
                                        youtuber_data.loc[youtuber_data.nome == youtuber, 'videosTranscritos'] += 1
                                        youtuber_data.to_csv(youtuberListPath, index=False)
                                    except Exception as e:
                                        console.log(f"[red]ERRO[/]: {e}",log_locals=True)

def transcript_all_videos(model):
    """
    Funcao para realizar o speech-to-text em todos os videos coletados e com áudios baixado
    model -- qual modelo do whisper a ser utilizado (entrar na documentacao do whisper para ver opcoes)
    """
    base_dir = "files"
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        transcript_videos_youtuber(model, ytb_folder)
def main():
    #console.rule("tira por tempo")
    #gerar_tira(60,"")
    #console.rule("tira por frase")
    #gerar_frases("files/OEPkmsJmY2I_text_small.json")
    #console.rule("tira por tempo e frase")
    # download_videos_youtuber('AuthenticGames')
    # transcript_videos_youtuber('tiny','AuthenticGames')
    download_videos_youtuber('Amy Scarlet')
    transcript_videos_youtuber('tiny','Amy Scarlet')
    #gerar_tira_frase_tempo(60, "files/AuthenticGames/2023/Novembro/MEU AMIGO visitou MEU MUNDO de MINECRAFT! @spok/video_text.json")


if __name__ == "__main__":
    main()
