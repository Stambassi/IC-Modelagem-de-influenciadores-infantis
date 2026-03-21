import yt_dlp
import os
import json
import csv
import time
import pandas as pd
import whisper 
from rich.console import Console
from pathlib import Path
from typing import List

# -------------------------------------------------------------------
# CONFIGURAÇÕES GLOBAIS E MAPEAMENTO
# -------------------------------------------------------------------
console = Console(color_system="auto")
CSV_TRANSCRIPTED = "transcripted_videos.csv"
YOUTUBER_LIST_PATH = "youtuberslist.csv"
BASE_DIR = Path("files")

MAPA_YOUTUBERS_CATEGORIA = {
    'Amy Scarlet': 'Roblox',
    'AuthenticGames': 'Minecraft',
    'Athos': 'Minecraft',
    'Cadres': 'Minecraft',
    'Julia MineGirl': 'Roblox',
    'Kass e KR': 'Minecraft',
    'Lokis': 'Roblox',
    'Luluca Games': 'Roblox',
    'Papile': 'Roblox',
    'Robin Hood Gamer': 'Minecraft',
    'TazerCraft': 'Minecraft',
    'Tex HS': 'Misto'
}

# -------------------------------------------------------------------
# 1. FUNÇÕES DE GERENCIAMENTO E SINCRONIZAÇÃO
# -------------------------------------------------------------------

'''
    Filtra a lista de youtubers com base em um critério (Geral, Categoria ou Nome Individual)

    @param nome_grupo - String com o nome do grupo, categoria ou canal desejado
'''
def obter_lista_youtubers(nome_grupo: str) -> List[str]:
    if nome_grupo == "Geral":
        return list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    categorias = set(MAPA_YOUTUBERS_CATEGORIA.values())
    if nome_grupo in categorias:
        return [y for y, cat in MAPA_YOUTUBERS_CATEGORIA.items() if cat == nome_grupo]
    
    if nome_grupo in MAPA_YOUTUBERS_CATEGORIA:
        return [nome_grupo]
    
    return []

'''
    Registra um vídeo como processado no arquivo de controle CSV para evitar redundância

    @param youtuber - Nome do canal do youtuber
    @param video_id - ID único do vídeo do YouTube
'''
def atualizar_status_csv(youtuber: str, video_id: str):
    nova_linha = pd.DataFrame({'nome': [youtuber], 'video_id': [video_id]})
    
    if not os.path.exists(CSV_TRANSCRIPTED):
        nova_linha.to_csv(CSV_TRANSCRIPTED, mode='w', header=True, index=False)
        return
        
    df = pd.read_csv(CSV_TRANSCRIPTED)
    if not ((df['nome'] == youtuber) & (df['video_id'] == video_id)).any():
        nova_linha.to_csv(CSV_TRANSCRIPTED, mode='a', header=False, index=False)

'''
    Verifica se um vídeo específico já consta no log de vídeos processados

    @param video_id - ID único do vídeo a ser verificado
'''
def video_ja_processado(video_id: str) -> bool:
    if not os.path.exists(CSV_TRANSCRIPTED): 
        return False
    df = pd.read_csv(CSV_TRANSCRIPTED)
    return str(video_id) in df['video_id'].astype(str).values

'''
    Conta quantos arquivos de transcrição existem para um youtuber e atualiza o dashboard principal

    @param youtuber - Nome do canal do youtuber
'''
def atualizar_video_total_transcritos(youtuber: str) -> int:
    base_dir = BASE_DIR / youtuber
    videos = 0
    if base_dir.exists():
        for json_path in base_dir.rglob('video_text.json'):
            if json_path.stat().st_size > 0:
                videos += 1
    
    if os.path.exists(YOUTUBER_LIST_PATH):
        df = pd.read_csv(YOUTUBER_LIST_PATH)
        df.loc[df.nome == youtuber, 'videosTranscritos'] = videos
        df.to_csv(YOUTUBER_LIST_PATH, index=False)
        
    return videos

# -------------------------------------------------------------------
# 2. LÓGICA DE PROCESSAMENTO (WHISPER E DOWNLOAD)
# -------------------------------------------------------------------

'''
    Realiza o download do áudio de um vídeo do YouTube e converte para MP3 192kbps

    @param video_id - ID único do vídeo para download
    @param output_folder - Caminho da pasta onde o áudio será salvo
'''
def acao_baixar_audio(video_id: str, output_folder: str) -> str:
    audio_path = os.path.join(output_folder, f"{video_id}.mp3")
    
    if os.path.exists(audio_path): 
        return audio_path
        
    console.print(f"> [yellow]Baixando audio[/] | video_id({video_id})")
    
    ydl_opts = {
        # 'bestaudio' busca a melhor qualidade de áudio independente do container
        'format': 'bestaudio/best',
        'outtmpl': f'{output_folder}/%(id)s.%(ext)s',
        # Noplaylist garante que não tente baixar uma playlist inteira se o ID vier de uma
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True, 
        'no_warnings': True,
        # Adicionar cabeçalhos de navegador ajuda a evitar blocos e erros de formato
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Tenta extrair as informações antes para validar a existência
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        console.print(f"> Download do audio foi um [green]sucesso[/] | video_id({video_id})")
        return audio_path
    except Exception as e:
        console.print(f"   └── [red]Erro download: {str(e).split(';')[0]}[/red]")
        return None

'''
    Executa a transcrição do áudio via Whisper, salva o JSON e remove o arquivo de áudio

    @param audio_path - Caminho do arquivo MP3 local
    @param output_folder - Pasta de destino para o arquivo video_text.json
    @param model_obj - Objeto do modelo Whisper carregado na memória
    @param video_id - ID do vídeo para registro no log
    @param youtuber - Nome do canal para registro no log
'''
def acao_transcrever_audio(audio_path: str, output_folder: str, model_obj, video_id: str, youtuber: str):
    json_path = os.path.join(output_folder, "video_text.json")
    if os.path.exists(json_path): 
        return
    
    try:
        start_time = time.time()
        with console.status("[cyan]Transcrevendo audio...", spinner="dots", refresh_per_second=5.0, speed=0.5):
            transcricao = model_obj.transcribe(audio_path, language='pt')
        
        with open(json_path, mode='w', encoding='utf-8') as file:
            json.dump(transcricao, file, ensure_ascii=False, indent=4)
        
        console.print("> Transcrição feita com [green] sucesso [/]")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
            console.print("> Áudio deletado com [green] sucesso [/]")
            
        atualizar_status_csv(youtuber, video_id)
        
        execution_time = time.time() - start_time
        console.print(f">>> Tempo de execução do Video_id ({video_id}) foi de [red]{execution_time:.2f} segundos [/] [gray]({execution_time/60:.2f} minutos)[/]")
        
    except Exception as e:
        console.log(f"[red] Erro [/] ao processar o áudio: {e}", log_locals=True)

# -------------------------------------------------------------------
# 3. FUNÇÕES DE ANÁLISE E UTILITÁRIOS
# -------------------------------------------------------------------

'''
    Converte o dicionário de transcrição do Whisper em um arquivo CSV detalhado por segmentos

    @param data - Dicionário retornado pela transcrição do Whisper
    @param output_folder - Pasta onde o arquivo CSV será gerado
    @param video_id - ID do vídeo para nomeação do arquivo
'''
def result_to_csv(data: dict, output_folder: str, video_id: str):
    console.print(f"> Criando CSV | path: {output_folder}")
    csv_file = os.path.join(output_folder, f"{video_id}_text_small.csv")
    headers = ['id', 'seek', 'start', 'end', 'text', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob']

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for segment in data['segments']:
            writer.writerow({k: segment.get(k) for k in headers})
    console.print(f"CSV file '{csv_file}' has been created.")

'''
    Exibe uma lista de tiras de texto numeradas no console para conferência

    @param tiras - Lista de strings contendo as tiras processadas
'''
def show_tiras(tiras: List[str]):
    for x, tira in enumerate(tiras):
        print(f"{x}: {tira}\n")

'''
    Realiza o agrupamento bruto de segmentos da transcrição em blocos fixos de tempo

    @param tempo - Intervalo de tempo desejado em segundos para cada tira
    @param data_path - Caminho do arquivo video_text.json
'''
def gerar_tira(tempo: int, data_path: str):
    margem = 10
    tempo_real = tempo * (1 - (margem / 100))
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        total_time, tira_atual, tiras = 0, "", []
        for segment in data["segments"]:
            total_time += (segment['end'] - segment['start'])
            tira_atual += segment['text'] 
            if total_time >= tempo_real:
                tiras.append({"text": tira_atual, "time": total_time})
                tira_atual, total_time = "", 0
        if tira_atual:
            tiras.append({"text": tira_atual, "time": total_time})
    
    for x, i in enumerate(tiras):
        print(f"{x}: {i['text']} [{i['time']:.2f}s]\n")
    console.print(f"Total de tiras: {len(tiras)}")

'''
    Agrupa os segmentos de transcrição baseando-se estritamente na ocorrência de pontos finais

    @param data_path - Caminho do arquivo video_text.json
'''
def gerar_frases(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)        
        tira_atual, tiras = "", []
        for segment in data["segments"]:
            tira_atual += segment['text'] 
            if tira_atual.strip().endswith('.'):
                tiras.append(tira_atual.strip())
                tira_atual = ""
        if tira_atual: tiras.append(tira_atual.strip())
        show_tiras(tiras)
        console.print(f"Total de tiras: {len(tiras)}")

'''
    Lógica avançada de segmentação: agrupa por tempo mínimo, mas só corta a tira ao encontrar pontuação

    @param tempo - Tempo base em segundos para a janela de corte
    @param data_path - Objeto Path apontando para o arquivo video_text.json
'''
def gerar_tira_frase_tempo(tempo: int, data_path: Path) -> List[str]:
    margem = 10
    tempo_real = tempo * (1 - (margem / 100))
    if not data_path.exists(): return []
    
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        total_time, tira_atual, tiras = 0, "", []
        for segment in data["segments"]:
            total_time += (segment['end'] - segment['start'])
            tira_atual += " " + segment['text']
            if total_time >= tempo_real:
                i = max(tira_atual.rfind("."), tira_atual.rfind("?"), tira_atual.rfind("!"))
                if i < 0: i = len(tira_atual)
                tiras.append(tira_atual[:i+1].strip())
                tira_atual, total_time = tira_atual[i+1:], 0
        if tira_atual.strip(): tiras.append(tira_atual.strip())
        return tiras

'''
    Gera o arquivo tiras_video.csv para um vídeo específico utilizando a lógica de tempo e frase

    @param output_folder - Pasta onde o vídeo está armazenado
    @param tempo_segundos - Tamanho da janela temporal em segundos (padrão 60s)
'''
def acao_dividir_em_tiras(output_folder: str, tempo_segundos: int = 60):
    json_path = Path(output_folder) / "video_text.json"
    tiras = gerar_tira_frase_tempo(tempo_segundos, json_path)
    if tiras:
        pd.DataFrame(tiras, columns=['tiras']).to_csv(Path(output_folder) / "tiras_video.csv", index_label='index')
        console.print(f"   └── [cyan]Tiras geradas: {len(tiras)}[/]")

# -------------------------------------------------------------------
# 4. ORQUESTRAÇÃO
# -------------------------------------------------------------------

'''
    Percorre recursivamente as pastas de um youtuber para aplicar uma ação em cada vídeo encontrado

    @param youtuber - Nome do canal do youtuber
    @param acao_escolhida - Tipo de processamento ('baixar', 'transcrever', 'dividir', etc.)
    @param model_obj - Modelo Whisper carregado (opcional, apenas para transcrição)
'''
def processar_diretorios(youtuber: str, acao_escolhida: str, model_obj=None):
    base_dir = BASE_DIR / youtuber
    if not base_dir.exists(): return
    
    console.rule(f"[bold red]Youtuber: {youtuber}")
    
    for csv_path in base_dir.rglob("videos_info.csv"):
        folder_path = csv_path.parent
        try:
            df = pd.read_csv(csv_path)
            video_id = str(df.iloc[0]['video_id'])
            video_folder_name = folder_path.name

            console.print(f"[bold cyan]>>> Processando Video:[/] {youtuber} ({video_folder_name})")
            
            if acao_escolhida == 'baixar':
                if not video_ja_processado(video_id): 
                    acao_baixar_audio(video_id, str(folder_path))
                else:
                    console.print("[i]Video ja coletado![/] Passando para o proximo...")
                    
            elif acao_escolhida == 'transcrever':
                if video_ja_processado(video_id):
                    console.print("[i]Video ja transcrito![/] Passando para o proximo...")
                    continue
                
                audio_path = os.path.join(folder_path, f"{video_id}.mp3")
                if not os.path.exists(audio_path):
                    audio_path = acao_baixar_audio(video_id, str(folder_path))
                
                if audio_path: 
                    acao_transcrever_audio(audio_path, str(folder_path), model_obj, video_id, youtuber)
                    
            elif acao_escolhida == 'transcrever_local':
                audio_path = os.path.join(folder_path, f"{video_id}.mp3")
                if os.path.exists(audio_path): 
                    acao_transcrever_audio(audio_path, str(folder_path), model_obj, video_id, youtuber)
                    
            elif acao_escolhida == 'dividir': 
                acao_dividir_em_tiras(str(folder_path))
                
        except Exception as e:
            console.print(f"[red]Erro na pasta {folder_path}: {e}[/red]")

'''
    Ponto de entrada que gerencia a carga do modelo e o loop de execução entre múltiplos influenciadores

    @param grupo_alvo - Critério de seleção ('Geral', categoria ou nome individual)
    @param acao - Nome da ação a ser executada em massa
    @param nome_modelo - Tamanho do modelo Whisper a ser carregado (tiny, base, small, etc.)
'''
def orquestrar_processamento(grupo_alvo: str, acao: str, nome_modelo: str = "tiny"):
    lista = obter_lista_youtubers(grupo_alvo)
    if not lista: 
        console.print("[red]Alvo não encontrado.[/red]")
        return
        
    model_obj = None
    if 'transcrever' in acao:
        console.print(f"[yellow]Carregando modelo {nome_modelo}...[/]")
        model_obj = whisper.load_model(nome_modelo)
        
    for ytb in lista: 
        processar_diretorios(ytb, acao, model_obj)
        atualizar_video_total_transcritos(ytb)

if __name__ == "__main__":
    pass