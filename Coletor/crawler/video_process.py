import yt_dlp
import os
import json
import csv
import time
import pandas as pd
import whisper 
from rich.console import Console
from pathlib import Path

# Configurações globais
console = Console(color_system="auto")
CSV_TRANSCRIPTED = "transcripted_videos.csv"
YOUTUBER_LIST_PATH = "youtuberslist.csv"
BASE_DIR = Path("files")

# Mapa de categorias
MAPA_YOUTUBERS_CATEGORIA = {
    'Amy Scarlet': 'Roblox',
    'AuthenticGames': 'Minecraft',
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
# 1. FUNÇÕES AUXILIARES DE GERENCIAMENTO
# -------------------------------------------------------------------

'''
    Retorna a lista de youtubers com base no filtro solicitado
    
    @param nome_grupo - Nome do grupo ('Geral', categoria ou nome do youtuber)
    @return lista - Lista de strings com nomes dos youtubers
'''
def obter_lista_youtubers(nome_grupo):
    if nome_grupo == "Geral":
        return list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    # Verifica se é uma categoria
    categorias = set(MAPA_YOUTUBERS_CATEGORIA.values())
    if nome_grupo in categorias:
        return [y for y, cat in MAPA_YOUTUBERS_CATEGORIA.items() if cat == nome_grupo]
    
    # Verifica se é um youtuber individual
    if nome_grupo in MAPA_YOUTUBERS_CATEGORIA:
        return [nome_grupo]
        
    console.print(f"[bold red]Aviso:[/] Grupo '{nome_grupo}' não encontrado. Retornando vazio.")
    return []

'''
    Atualiza o CSV de controle (transcripted_videos.csv) marcando o vídeo como processado
    
    @param youtuber - Nome do youtuber
    @param video_id - ID do vídeo processado
    @return None
'''
def atualizar_status_csv(youtuber, video_id):
    nova_linha = pd.DataFrame({'nome': [youtuber], 'video_id': [video_id]})
    
    if not os.path.exists(CSV_TRANSCRIPTED):
        nova_linha.to_csv(CSV_TRANSCRIPTED, mode='w', header=True, index=False)
        return

    df = pd.read_csv(CSV_TRANSCRIPTED)
    
    # Verifica se já existe para não duplicar
    if not ((df['nome'] == youtuber) & (df['video_id'] == video_id)).any():
        nova_linha.to_csv(CSV_TRANSCRIPTED, mode='a', header=False, index=False)

'''
    Verifica se o vídeo já consta no CSV de processados.
    
    @param video_id - ID do vídeo
    @return bool - True se já foi processado
'''
def video_ja_processado(video_id):
    if not os.path.exists(CSV_TRANSCRIPTED):
        return False
        
    df = pd.read_csv(CSV_TRANSCRIPTED)
    return video_id in df['video_id'].values

# -------------------------------------------------------------------
# 2. AÇÕES PRINCIPAIS (BAIXAR, TRANSCREVER, DIVIDIR)
# -------------------------------------------------------------------

'''
    Realiza o download do áudio do vídeo via yt_dlp
    
    @param video_id - ID do vídeo do YouTube
    @param output_folder - Caminho da pasta onde salvar
    @return audio_path - Caminho completo do arquivo baixado ou None em caso de erro
'''
def acao_baixar_audio(video_id, output_folder):
    audio_path = os.path.join(output_folder, f"{video_id}.mp3")
    
    if os.path.exists(audio_path):
        return audio_path

    console.print(f"   └── [yellow]Baixando áudio...[/] ({video_id})")
    
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
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        console.print("   └── [green]Download concluído![/]")
        return audio_path
    except Exception as e:
        console.print(f"   └── [red]Erro no download: {e}[/red]")
        return None

'''
    Transcreve o arquivo de áudio usando o modelo Whisper carregado
    
    @param audio_path - Caminho do arquivo mp3
    @param output_folder - Pasta para salvar o JSON
    @param model_obj - Objeto do modelo Whisper já carregado
    @param video_id - ID do vídeo
    @param youtuber - Nome do youtuber para registro
    @return dict - Resultado da transcrição ou None
'''
def acao_transcrever_audio(audio_path, output_folder, model_obj, video_id, youtuber):
    json_path = os.path.join(output_folder, "video_text.json")
    
    if os.path.exists(json_path):
        console.print("   └── [blue]JSON já existe. Pulando transcrição.[/]")
        return None

    if not os.path.exists(audio_path):
        console.print("   └── [red]Áudio não encontrado para transcrição.[/]")
        return None

    try:
        start_time = time.time()
        # Força português para evitar alucinações em inglês
        transcricao = model_obj.transcribe(audio_path, language='pt')
        
        with open(json_path, mode='w', encoding='utf-8') as file:
            json.dump(transcricao, file, ensure_ascii=False, indent=4)
            
        # Limpeza: Deleta áudio após sucesso
        os.remove(audio_path)
        
        # Atualiza registros
        atualizar_status_csv(youtuber, video_id)
        
        tempo = time.time() - start_time
        console.print(f"   └── [green]Transcrito em {tempo:.2f}s e áudio deletado.[/]")
        return transcricao
        
    except Exception as e:
        console.print(f"   └── [red]Erro na transcrição: {e}[/red]")
        return None

'''
    Lê o JSON de transcrição e cria um CSV dividido por janelas de tempo (tiras)

    @param output_folder - Pasta onde está o video_text.json
    @param tempo_segundos - Tamanho da janela de tempo para cada tira
    @return None
'''
def acao_dividir_em_tiras(output_folder, tempo_segundos=60):
    json_path = os.path.join(output_folder, "video_text.json")
    csv_tiras_path = os.path.join(output_folder, "tiras_video.csv")
    
    if not os.path.exists(json_path):
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        tiras = []
        tira_atual = ""
        tempo_acumulado = 0
        
        # Margem de segurança de 10%
        tempo_alvo = tempo_segundos * 0.9 
        
        for segment in data["segments"]:
            duracao = segment['end'] - segment['start']
            texto = segment['text'].strip()
            
            tira_atual += " " + texto
            tempo_acumulado += duracao
            
            if tempo_acumulado >= tempo_alvo:
                tiras.append(tira_atual.strip())
                tira_atual = ""
                tempo_acumulado = 0
                
        # Adiciona o resto, se houver
        if tira_atual:
            tiras.append(tira_atual.strip())
            
        # Salva em CSV
        df_tiras = pd.DataFrame(tiras, columns=['tiras'])
        df_tiras.to_csv(csv_tiras_path, index=False)
        console.print(f"   └── [cyan]Tiras geradas: {len(tiras)} (Salvo em tiras_video.csv)[/]")
        
    except Exception as e:
        console.print(f"   └── [red]Erro ao gerar tiras: {e}[/red]")

# -------------------------------------------------------------------
# 3. ORQUESTRADOR
# -------------------------------------------------------------------

'''
    Navega pela estrutura de pastas (Files -> Youtuber -> Ano -> Mês -> Video) e executa a ação
    
    @param youtuber - Nome do youtuber
    @param acao_escolhida - 'baixar', 'transcrever' ou 'dividir'
    @param model_obj - Modelo Whisper (apenas se acao == 'transcrever')
    @return None
'''
def processar_diretorios(youtuber, acao_escolhida, model_obj=None):
    base_dir = BASE_DIR / youtuber
    if not base_dir.exists():
        console.print(f"[red]Diretório não encontrado: {base_dir}[/]")
        return

    console.print(f"\n[bold magenta]Processando Youtuber: {youtuber}[/]")
    
    # Navegação recursiva: Files/Youtuber/Ano/Mes/Video
    # Usando rglob para achar todos os 'videos_info.csv' que marcam a pasta de um vídeo
    for csv_path in base_dir.rglob("videos_info.csv"):
        folder_path = str(csv_path.parent)
        
        try:
            df = pd.read_csv(csv_path)
            if df.empty: continue
            video_id = str(df.iloc[0]['video_id'])
            
            console.print(f"• Vídeo: {video_id}", style="dim")

            # --- SELETOR DE AÇÃO ---
            if acao_escolhida == 'baixar':
                # Só baixa se não tiver transcrito ainda
                if not video_ja_processado(video_id):
                    acao_baixar_audio(video_id, folder_path)
                else:
                    console.print("   └── [green]Já transcrito. Pula download.[/]")

            elif acao_escolhida == 'transcrever':
                audio_path = os.path.join(folder_path, f"{video_id}.mp3")
                # Se o áudio não existe, tenta baixar primeiro
                if not os.path.exists(audio_path) and not video_ja_processado(video_id):
                     audio_path = acao_baixar_audio(video_id, folder_path)
                
                if audio_path:
                    acao_transcrever_audio(audio_path, folder_path, model_obj, video_id, youtuber)

            elif acao_escolhida == 'dividir':
                acao_dividir_em_tiras(folder_path)

        except Exception as e:
            console.print(f"[red]Erro crítico na pasta {folder_path}: {e}[/red]")


'''
    Função Principal (Controller) que gerencia o fluxo de execução.
    
    @param grupo_alvo - 'Geral', Categoria (ex: 'Minecraft') ou Nome do Youtuber
    @param acao - 'baixar', 'transcrever', 'dividir'
    @param nome_modelo - Modelo do Whisper ('tiny', 'base', 'medium')
    @return None
'''
def orquestrar_processamento(grupo_alvo, acao, nome_modelo="tiny"):
    lista_youtubers = obter_lista_youtubers(grupo_alvo)
    
    if not lista_youtubers:
        return

    console.rule(f"[bold]Iniciando Ação: {acao.upper()} | Grupo: {grupo_alvo}[/bold]")

    # Carrega modelo apenas uma vez e se necessário
    model_obj = None
    if acao == 'transcrever':
        console.print(f"[yellow]Carregando modelo Whisper ({nome_modelo})... aguarde.[/yellow]")
        try:
            model_obj = whisper.load_model(nome_modelo)
            console.print("[green]Modelo carregado![/green]")
        except Exception as e:
            console.print(f"[red]Falha ao carregar modelo: {e}[/red]")
            return

    # Itera sobre os membros do grupo
    for youtuber in lista_youtubers:
        processar_diretorios(youtuber, acao, model_obj)

    console.rule("[bold green]Processamento Finalizado[/bold green]")

# -------------------------------------------------------------------
# 4. MAIN
# -------------------------------------------------------------------

'''
    Ponto de entrada do script.
    Opções de Ação: 'baixar', 'transcrever', 'dividir'
    Opções de Grupo: 'Geral', 'Minecraft', 'Roblox', 'NomeYoutuber'
'''
if __name__ == "__main__": 
    # Exemplo 1: Baixar e Transcrever apenas Minecraft
    # orquestrar_processamento(grupo_alvo='Minecraft', acao='transcrever', nome_modelo='medium')

    # Exemplo 2: Apenas dividir em tiras para um Youtuber específico
    #orquestrar_processamento(grupo_alvo='Julia MineGirl', acao='dividir')

    # Exemplo 3: Baixar áudios faltantes de todos (sem transcrever)
    # orquestrar_processamento(grupo_alvo='Geral', acao='baixar')

    orquestrar_processamento('Geral', 'dividir')