import yt_dlp
import os
import wave
import json
import subprocess
import time
from rich.console import Console

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
        console.log("[red] Erro [/] ao processar o áudio: ", log_locals=True)
        print(e)
        return None


def video_to_text(video_id, output_folder):
    start_time = time.time()
    local_audio = download_youtube_audio(video_id, output_folder)
    model_directory = "vosk-model-small-pt-0.3"
    transcription_result = process_and_delete_audio(local_audio, model_directory)
    execution_time = time.time() - start_time
    console.print(">>> Tempo de execução do Video_id ("+video_id+") foi de [red]"+str(execution_time)+" segundos [/]")

    return transcription_result


