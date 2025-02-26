import yt_dlp
import os
import wave
import json
import csv
import subprocess
import time
import pandas as pd

from rich.console import Console

from vosk import Model, KaldiRecognizer

console = Console()


# Codigo salvo antes da alteracao completa para o whisper (Pode nao estar funcionando)

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
        'quiet': True, 
        'no_warnings': True,  
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    console.print("> Download do audio foi um [green]sucesso[/] | video_id("+video_id+")")
    files = f"{folder}/{video_id}"
    input_file = files + ".mp3"
    output_file = files + ".wav"
    convert_to_wav(input_file, output_file)
    return files


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
        # print(f"> Transcrevendo o audio | arquivo: {file_path}")
        text = transcribe_audio(file_path, model_path) 
        # Exibe o resultado
        console.print("> Transcrição feita com [green] sucesso [/]")
        # print(text)
        
        # Remove o arquivo de áudio após a transcrição
        os.remove(file_path+".mp3")
        os.remove(file_path+".wav")
        console.print(f"> Arquivo deletado com [green] sucesso [/]")
        
        return text
    except Exception as e:
        console.log("[red] Erro [/] ao processar o áudio: ", log_locals=False)
        print(e)
        return None



def video_to_text(video_id, output_folder, model, youtuber):
    start_time = time.time()
    local_audio = download_youtube_audio(video_id, output_folder)
    model_directory = "vosk-model-medium-pt-0.3"
    transcription_result = process_and_delete_audio(local_audio, model_directory)  
    execution_time = time.time() - start_time
    console.print(">>> Tempo de execução do Video_id ("+video_id+") foi de [red]"+str(execution_time)+" segundos [/] [gray]("+str(execution_time/60)+" minutos)[/]")

    return transcription_result

def process_video(csv, file_path, model, youtuber):
    df = pd.read_csv(csv)
    video_id = df.loc[0]['video_id']
    return video_to_text(video_id, file_path, model, youtuber)

def process_youtuber_video(model, youtuber):
    base_dir = f"files/{youtuber}"
    if os.path.isdir(base_dir):
        console.rule("[bold red]Youtuber: "+youtuber)
        # andar pelos anos
        for year_folder in os.listdir(base_dir):
            next_year_dir = os.path.join(base_dir, year_folder)
            if os.path.isdir(next_year_dir):
                # andar pelos meses
                for month_folder in os.listdir(next_year_dir):
                    console.log("[bold cyan]"+youtuber+" ("+month_folder+"/"+year_folder+"): ")
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
                                    text = process_video(csv_path,folder_path, model, youtuber)
                                    print(text['text'])

def process_all_videos(model):
    base_dir = "files"
    # andar pelos youtubers
    for ytb_folder in os.listdir(base_dir):
        process_youtuber_video(model,ytb_folder)

def main():
    # video_to_text("OEPkmsJmY2I","files")
    # video_to_text("PXHlV3g4lpg","files")
    process_all_videos("tiny")
    # process_youtuber_video("tiny","TazerCraft")

if __name__ == "__main__":
    main()