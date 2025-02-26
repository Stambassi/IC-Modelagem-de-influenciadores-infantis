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
from rich.json import JSON

from vosk import Model, KaldiRecognizer

console = Console()
def convert_to_wav(input_file, output_file):
    print(f"> Convertendo para WAV | Arquivo ({input_file})")
    try:
        command = [
            "ffmpeg", "-i", input_file,
            "-ac", "1", "-ar", "16000", output_file
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"> Sucesso na conversao para WAV: {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Erro ao converter áudio: {e}")


def download_youtube_audio(video_id, output_folder):
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
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    console.print("> Download do audio foi um [green]sucesso[/] | video_id("+video_id+")")
    files = f"{folder}/{video_id}"
    input_file = files + ".mp3"
    output_file = files + ".wav"
    convert_to_wav(input_file, output_file)
    return files

def transcribe_audio_whisper(file_path, model):
    with console.status("Transcrevendo audio...",spinner="aesthetic",refresh_per_second=5.0,speed=0.1):
        modelo = whisper.load_model(model) #, devide = "cpu" // para rodar usando a cpu
        full_file = file_path + ".mp3"
        resposta = modelo.transcribe(full_file)
        return resposta

def transcribe_audio(file_path, model_path):
    # Verifica se o modelo está disponível
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Faça o download em https://alphacephei.com/vosk/models e descompacte.")
    
    # Carrega o modelo Vosk
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    full_file = file_path + ".wav"
    # Abre o arquivo de áudio
    with wave.open(full_file, "rb") as wf:
        # Verifica se o arquivo de áudio é mono e tem frequência de 16 kHz
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            raise ValueError("O arquivo de áudio precisa ser mono e ter uma taxa de amostragem de 16 kHz.")
        
        recognizer.SetWords(True)
        transcription = []
        with console.status("[bold green]Processando video...") as status:
        # Lê o áudio em blocos e realiza a transcrição
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcription.append(result.get("text", ""))
        
        # Obtém o texto final
        final_result = json.loads(recognizer.FinalResult())
        transcription.append(final_result.get("text", ""))
    # Retorna o texto completo
    return " ".join(transcription)

# Função principal para transcrever e deletar o arquivo
def process_and_delete_audio(file_path, model_path="model"):
    try:
        # Realiza a transcrição
        print(f"> Transcrevendo o audio | arquivo: {file_path}")
        text = transcribe_audio(file_path, model_path) 
        # Exibe o resultado
        console.print("> Transcrição feita com [green] sucesso [/] | arquivo: "+file_path)
        # print(text)
        
        # Remove o arquivo de áudio após a transcrição
        os.remove(file_path+".mp3")
        os.remove(file_path+".wav")
        console.print(f"> Arquivo deletado com [green] sucesso [/] | arquivo: "+file_path)
        
        return text
    except Exception as e:
        console.log("[red] Erro [/] ao processar o áudio: ", log_locals=False)
        print(e)
        return None

def process_and_delete_audio_whisper(file_path, model):
    try:
        # Realiza a transcrição
        print(f"> Transcrevendo o audio | arquivo: {file_path}")
        text = transcribe_audio_whisper(file_path, model) 
        # Exibe o resultado
        console.print("> Transcrição feita com [green] sucesso [/] | arquivo: "+file_path)
        print(text['text']) 
        # data = pd.DataFrame(text)
        # p = "files/speech_to_text_Whisper.csv"
        # # print(data)
        # data.to_csv(p)
        
        # Remove o arquivo de áudio após a transcrição
        os.remove(file_path+".mp3")
        os.remove(file_path+".wav")
        console.print(f"> Arquivo deletado com [green] sucesso [/] | arquivo: "+file_path)
        # console.print("[bold cyan]Fazendo testes sem deletar arquivos de audio")
        
        return text
    except Exception as e:
        console.log("[red] Erro [/] ao processar o áudio: ", log_locals=False)
        print(e)
        return None

def result_to_csv(data,output_folder,id):
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



def video_to_text(video_id, output_folder, model):
    start_time = time.time()
    local_audio = download_youtube_audio(video_id, output_folder)
    # local_audio = f"{output_folder}/{video_id}"
    # model_directory = "vosk-model-medium-pt-0.3"
    # transcription_result = process_and_delete_audio(local_audio, model_directory)
    transcription_result = process_and_delete_audio_whisper(local_audio, model)
    json_path = f"{output_folder}/video_text.json"
    with open(json_path, mode='w', encoding='utf-8') as file:
        json.dump(transcription_result, file, ensure_ascii=False, indent=4)
    # result_to_csv(transcription_result,output_folder,video_id)

    execution_time = time.time() - start_time
    console.print(">>> Tempo de execução do Video_id ("+video_id+") foi de [red]"+str(execution_time)+" segundos [/] [gray]("+str(execution_time/60)+" minutos)[/]")

    return transcription_result

def process_video(csv, file_path, model):
    df = pd.read_csv(csv)
    video_id = df.loc[0]['video_id']
    return video_to_text(video_id, file_path, model)

def process_all_videos():
    base_dir = "files"
    console = Console()

    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        next_ytb_dir = os.path.join(base_dir, ytb_folder)
        if os.path.isdir(next_ytb_dir):
            # andar pelos anos
            for year_folder in os.listdir(next_ytb_dir):
                next_year_dir = os.path.join(next_ytb_dir, year_folder)
                if os.path.isdir(next_year_dir):
                    # andar pelos meses
                    for month_folder in os.listdir(next_year_dir):
                        next_month_dir = os.path.join(next_year_dir, month_folder)
                        if os.path.isdir(next_month_dir):
                        # andar pelos videos
                            for folder in os.listdir(next_month_dir):
                                folder_path = os.path.join(next_month_dir, folder)
                                if os.path.isdir(folder_path):
                                    csv_path = os.path.join(folder_path, 'videos_info.csv')
                                    # Check if the file exists
                                    if os.path.exists(csv_path):
                                        # result_df = coletar_dados(csv_path=csv_path, folder_path=folder_path)
                                        text = process_video(csv_path,folder_path, "tiny")
                                        print(text['text'])

def main():
    # video_to_text("OEPkmsJmY2I","files")
    # video_to_text("PXHlV3g4lpg","files")
    process_all_videos()

if __name__ == "__main__":
    main()