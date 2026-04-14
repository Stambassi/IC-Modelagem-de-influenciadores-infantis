import json
import os
import shutil
import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from rich.console import Console
from rich.rule import Rule

from datetime import datetime
from config import config  # Importa os seus limites atuais (2020 a 2023)
import csv

# Garante resultados determinísticos
DetectorFactory.seed = 0
console = Console()

# Configuração
BASE_DIR = Path("files") 

MAPA_MESES_REV = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# Mapeamento Global de Youtubers e suas palavras-chave de escopo
MAPA_YOUTUBERS_CATEGORIA = {
    'Julia MineGirl': 'Roblox',
    'Papile': 'Roblox',
    'Tex HS': 'Roblox',
    'Amy Scarlet': 'Roblox',
    'Luluca Games': 'Roblox',
    'meu nome é david': 'Roblox',
    'Lokis': 'Roblox',

    'Robin Hood Gamer': 'Minecraft',
    'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft',
    'Athos': 'Minecraft',
    'JP Plays': 'Minecraft',
    'Marcelodrv': 'Minecraft',
    'Geleia': 'Minecraft',
    'Kass e KR': 'Minecraft',
}

def limpar_pastas_duplicadas(remover_arquivos: bool = False):
    console.rule(f"[bold blue]Buscando Pastas Duplicadas (Remoção: {remover_arquivos})[/bold blue]")
    
    mapa_videos = {}
    
    # Vasculha todos os diretórios buscando a identidade real do vídeo
    for root, dirs, files in os.walk(BASE_DIR):
        if "videos_info.csv" in files:
            caminho_pasta = Path(root)
            try:
                df_info = pd.read_csv(caminho_pasta / "videos_info.csv", dtype=str)
                if not df_info.empty and 'video_id' in df_info.columns:
                    vid = str(df_info.iloc[0]['video_id']).strip()
                    if vid not in mapa_videos:
                        mapa_videos[vid] = []
                    mapa_videos[vid].append(caminho_pasta)
            except:
                pass
    
    pastas_removidas = 0
    
    # Analisa as duplicatas
    for vid, pastas in mapa_videos.items():
        if len(pastas) > 1:
            # O padrão novo sempre tem o ID entre colchetes no nome da pasta
            pastas_novas = [p for p in pastas if f"[{vid}]" in p.name]
            pastas_antigas = [p for p in pastas if f"[{vid}]" not in p.name]
            
            # Só remove a antiga se a nova existir para assumir o lugar
            if pastas_novas and pastas_antigas:
                for pasta_velha in pastas_antigas:
                    acao = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]IDENTIFICADO (Modo Teste)[/bold yellow]"
                    console.print(f"{acao} Pasta Obsoleta: {pasta_velha.relative_to(BASE_DIR)}")
                    
                    if remover_arquivos:
                        try:
                            shutil.rmtree(pasta_velha)
                            pastas_removidas += 1
                            console.print("   └── [green]Pasta física deletada com sucesso.[/green]")
                        except Exception as e:
                            console.print(f"   └── [red]Erro ao deletar pasta: {e}[/red]")
                            
    console.print(f"\nTotal de pastas obsoletas resolvidas: {pastas_removidas}\n")

def gerenciar_transcricoes_erradas(remover_arquivos: bool = False):
    total_analisado = 0
    arquivos_ingles = []
    
    acao_texto = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]IDENTIFICADO (Modo Teste)[/bold yellow]"
    
    console.rule(f"[bold magenta]Validando Idioma (Remoção: {remover_arquivos})[/bold magenta]")

    for json_path in BASE_DIR.rglob("video_text.json"):
        total_analisado += 1
        
        try:
            # Abre e lê o arquivo
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verifica se o JSON é um dicionário válido antes de usar o .get()
            if not isinstance(data, dict):
                idioma_detectado = "lixo_corrompido"
            else:
                # Agora é seguro extrair o texto
                texto = data.get('text', '')
                
                # Pula textos muito curtos (válidos, mas sem dados suficientes para detectar)
                if len(texto) < 50:
                    continue

                # Detecção de idioma
                try:
                    idioma_detectado = detect(texto)
                except LangDetectException:
                    idioma_detectado = "unknown"  

            # Testa se identificou o idioma errado ou um dado corrompido
            if idioma_detectado != 'pt':
                caminho_relativo = json_path.relative_to(BASE_DIR)
                console.print(f"{acao_texto} [{idioma_detectado}]: {caminho_relativo}")
                arquivos_ingles.append(caminho_relativo)

                if remover_arquivos:
                    try:
                        # Deleta a transcrição errada/corrompida
                        json_path.unlink()
                        console.print("   ├── [green]video_text.json deletado.[/green]")
                        
                        # Deleta as tiras atreladas a esse lixo
                        tiras_path = json_path.parent / "tiras_video.csv"
                        if tiras_path.exists():
                            tiras_path.unlink()
                            console.print("   └── [green]tiras_video.csv deletado (Limpeza profunda).[/green]")
                        else:
                            console.print("   └── [dim]Nenhum tiras_video.csv atrelado para deletar.[/dim]")
                    except Exception as e:
                        console.print(f"   └── [red]Erro ao deletar: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Erro ao processar {json_path}: {e}[/red]")

    console.rule("[bold]Resumo da Operação[/bold]")
    console.print(f"Total de arquivos verificados: {total_analisado}")
    console.print(f"Arquivos em outro idioma ou corrompidos: {len(arquivos_ingles)}")
    
    if remover_arquivos:
        console.print("[green]Limpeza profunda concluída! Você pode rodar seus coletores novamente.[/green]")
    else:
        console.print("[bold yellow]Modo de Teste.[/bold yellow] Mude 'remover_arquivos = True' no final do script para executar a faxina.")

def limpar_nome_arquivo(nome: str) -> str:
    if pd.isna(nome) or nome.strip().lower() == "nan":
        return "SemTitulo"
    return "".join([c for c in str(nome) if c.isalpha() or c.isdigit() or c in " .-_"]).strip()

def padronizar_nomes_pastas(remover_arquivos: bool = False):
    acao_texto = "[bold red]RENOMEANDO[/bold red]" if remover_arquivos else "[bold yellow]IDENTIFICADO (Modo Teste)[/bold yellow]"
    
    console.print(Rule(f"[orange]Padronizador de Pastas (Remover arquivos: {remover_arquivos})[/orange]"))
    console.print(f"[dim]Varrendo diretório base: {BASE_DIR}...[/dim]\n")

    pastas_para_analisar = []

    # Passo 1: Coletar todos os caminhos
    for root, dirs, files in os.walk(BASE_DIR):
        if "videos_info.csv" in files:
            pastas_para_analisar.append(root)

    total_analisado = len(pastas_para_analisar)
    pastas_corrigidas = 0
    pastas_ignoradas_conflito = 0

    # Passo 2: Analisar e renomear
    for pasta in pastas_para_analisar:
        nome_atual = os.path.basename(pasta)
        
        try:
            # Lê a identidade do vídeo
            path_info = os.path.join(pasta, "videos_info.csv")
            df_info = pd.read_csv(path_info, dtype=str)
            
            if df_info.empty or 'video_id' not in df_info.columns:
                continue
                
            video_id = str(df_info.iloc[0]['video_id']).strip()
            
            # Se o ID já está no nome da pasta, ela está correta
            if f"[{video_id}]" in nome_atual:
                continue
                
            # Descobre o título para montar o nome ideal
            titulo_bruto = df_info.iloc[0].get('title', 'SemTitulo')
            titulo_safe = limpar_nome_arquivo(titulo_bruto)
            
            nome_esperado = f"{titulo_safe} [{video_id}]" if titulo_safe else f"[{video_id}]"
            
            pasta_pai = os.path.dirname(pasta)
            caminho_esperado = os.path.join(pasta_pai, nome_esperado)
            
            # Avalia se a pasta com o nome correto já existe (evita esmagar dados)
            if os.path.exists(caminho_esperado):
                console.print(f"[magenta]⚠️ CONFLITO:[/magenta] A pasta [cyan]{nome_atual}[/cyan] está fora do padrão, mas a correta já existe!")
                console.print(f"   └── Sugestão: Rode seu script de 'limpar_pastas_duplicadas' para resolver isso.")
                pastas_ignoradas_conflito += 1
                continue
            
            # Executa a renomeação
            console.print(f"{acao_texto}: [dim]{nome_atual}[/dim] -> [green]{nome_esperado}[/green]")
            
            if remover_arquivos:
                os.rename(pasta, caminho_esperado)
                pastas_corrigidas += 1

        except Exception as e:
            console.print(f"[bold red]Erro ao processar a pasta {pasta}: {e}[/bold red]")

    # Relatório Final
    console.print("")
    console.print(Rule("Resumo da Operação"))
    console.print(f"Total de pastas de vídeos verificadas: [cyan]{total_analisado}[/cyan]")
    console.print(f"Pastas fora do padrão e sem conflito: [cyan]{pastas_corrigidas if remover_arquivos else 'Aguardando execução'}[/cyan]")
    
    if pastas_ignoradas_conflito > 0:
        console.print(f"Pastas ignoradas por já existir a versão correta: [yellow]{pastas_ignoradas_conflito}[/yellow]")
        
    if not remover_arquivos:
        console.print("\n[bold yellow]Isso foi apenas um teste![/bold yellow] Nenhuma pasta foi alterada.")
        console.print("Altere [cyan]padronizar_nomes_pastas(remover_arquivos=True)[/cyan] no final do script para aplicar as mudanças de verdade.")
    else:
        console.print("\n[bold green]Faxina concluída com sucesso![/bold green]")

def filtrar_por_data(data_inicio: str, data_fim: str, remover_arquivos: bool = False):
    console.print(Rule(f"[bold cyan]Filtrando por Data: {data_inicio} até {data_fim} (Remover: {remover_arquivos})[/bold cyan]"))
    
    try:
        # Converter as datas para o padrão de tempo contendo timezone (UTC)
        inicio_dt = pd.to_datetime(data_inicio, utc=True)
        # O .replace para garantir que vá até o final do último dia do limite superior
        fim_dt = pd.to_datetime(data_fim, utc=True).replace(hour=23, minute=59, second=59)
    except Exception as e:
        console.print(f"[bold red]Erro ao interpretar as datas fornecidas: {e}[/bold red]")
        return

    pastas_removidas = 0
    total_analisado = 0
    
    for root, dirs, files in os.walk(BASE_DIR):
        if "videos_info.csv" in files:
            caminho_pasta = Path(root)
            total_analisado += 1
            
            try:
                # Carregar o CSV como string para evitar conversões indesejadas, depois processa a data
                df_info = pd.read_csv(caminho_pasta / "videos_info.csv", dtype=str)
                
                if not df_info.empty and 'published_at' in df_info.columns:
                    pub_str = df_info.iloc[0]['published_at']
                    pub_dt = pd.to_datetime(pub_str, errors='coerce', utc=True)
                    
                    if pd.notna(pub_dt):
                        # Condição: se a data do vídeo é MENOR que o início ou MAIOR que o fim
                        if pub_dt < inicio_dt or pub_dt > fim_dt:
                            acao = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]FORA DO INTERVALO (Teste)[/bold yellow]"
                            data_legivel = pub_dt.strftime('%d/%m/%Y')
                            
                            console.print(f"{acao} Data: {data_legivel} | Pasta: {caminho_pasta.relative_to(BASE_DIR)}")
                            
                            if remover_arquivos:
                                try:
                                    shutil.rmtree(caminho_pasta)
                                    pastas_removidas += 1
                                except Exception as e:
                                    console.print(f"   └── [red]Erro ao deletar: {e}[/red]")
                            else:
                                pastas_removidas += 1
            except Exception as e:
                console.print(f"[red]Erro ao processar o CSV na pasta {caminho_pasta}: {e}[/red]")

    # Relatório Final
    console.print("")
    console.print(f"Total de vídeos avaliados: [cyan]{total_analisado}[/cyan]")
    if remover_arquivos:
        console.print(f"Pastas deletadas (fora do limite): [green]{pastas_removidas}[/green]")
    else:
        console.print(f"Vídeos encontrados fora do limite: [yellow]{pastas_removidas}[/yellow]")
        console.print("\n[bold yellow]Isso foi apenas um teste![/bold yellow] Nenhuma pasta foi deletada.")

def sincronizar_datas_dashboard():
    console.rule("[bold cyan]Iniciando Auditoria e Sincronização de Datas[/bold cyan]")
    
    csv_path = "youtuberslist.csv"
    if not os.path.exists(csv_path):
        console.print("[red]Arquivo youtuberslist.csv não encontrado.[/red]")
        return
        
    df_dashboard = pd.read_csv(csv_path)
    base_dir = Path("files")
    
    # Limites da nova coleta vindos do config.py
    inicio_limite = pd.to_datetime(f"{config['start_date'][0]}-{config['start_date'][1]:02d}-{config['start_date'][2]:02d}", utc=True)
    fim_limite = pd.to_datetime(f"{config['end_date'][0]}-{config['end_date'][1]:02d}-{config['end_date'][2]:02d}", utc=True).replace(hour=23, minute=59, second=59)
    
    for index, row in df_dashboard.iterrows():
        youtuber = row['nome']
        youtuber_dir = base_dir / youtuber
        
        # Padrão: começa do teto da coleta nova (Ex: Jan/2023)
        data_alvo = fim_limite 
        
        # Se a pasta do youtuber existe, procura o vídeo mais antigo DENTRO do novo limite
        if youtuber_dir.exists():
            datas_encontradas = []
            
            # Varre todos os videos_info.csv
            for info_path in youtuber_dir.rglob("videos_info.csv"):
                try:
                    df_video = pd.read_csv(info_path, dtype=str)
                    if not df_video.empty and 'published_at' in df_video.columns:
                        pub_dt = pd.to_datetime(df_video.iloc[0]['published_at'], errors='coerce', utc=True)
                        
                        if pd.notna(pub_dt):
                            # Só considera o que está DENTRO do novo intervalo de pesquisa
                            if inicio_limite <= pub_dt <= fim_limite:
                                datas_encontradas.append(pub_dt)
                except:
                    pass
            
            # Se achou vídeos no intervalo, a próxima busca começa a partir do MAIS ANTIGO para economizar queries
            if datas_encontradas:
                data_alvo = min(datas_encontradas)
                
        # 1. Cria/Atualiza o atual_date.csv do Youtuber
        youtuber_dir.mkdir(parents=True, exist_ok=True)
        caminho_atual_date = youtuber_dir / "atual_date.csv"
        
        with open(caminho_atual_date, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["year", "month", "day"])

            writer.writeheader()
            writer.writerow({
                "year": data_alvo.year,
                "month": data_alvo.month,
                "day": data_alvo.day
            })
            
        # 2. Atualiza o dashboard na memória do Pandas
        df_dashboard.at[index, 'ultimoAnoColetado'] = data_alvo.year
        df_dashboard.at[index, 'ultimoMesColetado'] = MAPA_MESES_REV.get(data_alvo.month, "Desconhecido")
        
        console.print(f"[green]✔ {youtuber}:[/green] Sincronizado para [bold]{data_alvo.strftime('%d/%m/%Y')}[/bold]")

    # Salva o dashboard atualizado
    df_dashboard.to_csv(csv_path, index=False)
    console.print("\n[bold blue]Sincronização concluída com sucesso![/bold blue] O youtuberslist.csv e os checkpoints locais estão 100% alinhados.")

'''
    Remove pastas de vídeos que não contêm a palavra-chave (minecraft ou roblox)
    no título ou na descrição, de acordo com o mapeamento do youtuber
'''
def filtrar_por_palavras_chave(remover_arquivos: bool = False):
    console.rule(f"[bold cyan]Filtrando por Palavras-chave no Título/Descrição (Remover: {remover_arquivos})[/bold cyan]")
    
    pastas_removidas = 0
    total_analisado = 0

    # Iterar apenas pelas pastas de primeiro nível dentro de 'files/' (que são os Youtubers)
    if not BASE_DIR.exists():
        console.print("[red]Diretório base 'files' não encontrado.[/red]")
        return

    for youtuber_folder in os.listdir(BASE_DIR):
        caminho_youtuber = BASE_DIR / youtuber_folder
        if not caminho_youtuber.is_dir(): 
            continue

        # Descobre a keyword alvo para este Youtuber (em minúsculo)
        # Usa .get() com um default None e trata isso na lógica abaixo.
        categoria = MAPA_YOUTUBERS_CATEGORIA.get(youtuber_folder)
        if not categoria:
            console.print(f"[yellow]Aviso: Youtuber '{youtuber_folder}' não mapeado em MAPA_YOUTUBERS_CATEGORIA. Ignorando filtro.[/yellow]")
            continue
            
        keyword_alvo = categoria.lower()

        # Agora varre os vídeos deste Youtuber
        for root, dirs, files in os.walk(caminho_youtuber):
            if "videos_info.csv" in files:
                caminho_pasta = Path(root)
                total_analisado += 1
                
                try:
                    df_info = pd.read_csv(caminho_pasta / "videos_info.csv", dtype=str)
                    if not df_info.empty:
                        # Extrai e formata os 3 campos de forma segura
                        titulo = str(df_info.iloc[0].get('title', '')).lower()
                        descricao = str(df_info.iloc[0].get('description', '')).lower()
                        tags = str(df_info.iloc[0].get('tags', '')).lower()
                        
                        # Pandas costuma transformar células vazias em string "nan", então limpamos isso
                        if titulo == 'nan': titulo = ''
                        if descricao == 'nan': descricao = ''
                        if tags == 'nan': tags = ''

                        # A palavra-chave alvo DEVE estar em pelo menos UM dos três campos
                        tem_no_titulo = keyword_alvo in titulo
                        tem_na_descricao = keyword_alvo in descricao
                        tem_nas_tags = keyword_alvo in tags

                        if not (tem_no_titulo or tem_na_descricao or tem_nas_tags):
                            acao = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]FORA DO ESCOPO (Teste)[/bold yellow]"
                            console.print(f"{acao} [Ausência de '{keyword_alvo}'] | Pasta: {caminho_pasta.relative_to(BASE_DIR)}")
                            
                            if remover_arquivos:
                                try:
                                    shutil.rmtree(caminho_pasta)
                                    pastas_removidas += 1
                                except Exception as e:
                                    console.print(f"   └── [red]Erro ao deletar: {e}[/red]")
                            else:
                                pastas_removidas += 1
                except Exception as e:
                    console.print(f"[red]Erro ao processar o CSV na pasta {caminho_pasta}: {e}[/red]")

    console.print("")
    console.print(f"Total de vídeos avaliados: [cyan]{total_analisado}[/cyan]")
    
    if remover_arquivos:
        console.print(f"Pastas deletadas (sem a palavra-chave): [green]{pastas_removidas}[/green]")
        console.print("\n[dim]Nota: Como arquivos físicos foram removidos, rodar 'sincronizar_datas_dashboard()' pode ser uma boa ideia para realinhar seu atual_date.csv.[/dim]")
    else:
        console.print(f"Vídeos encontrados sem a palavra-chave: [yellow]{pastas_removidas}[/yellow]")
        console.print("\n[bold yellow]Isso foi apenas um teste![/bold yellow] Nenhuma pasta foi deletada.")

'''
    Remove arquivos de áudio (.mp3) redundantes caso a transcrição correspondente já exista.
    Pode atuar no 'local' (pasta files) ou no 'remoto' (pasta data via Parquet).
'''
def expurgar_audios_redundantes(alvo: str = "local", remover_arquivos: bool = False):
    alvo = alvo.lower()
    console.rule(f"[bold red]Limpador de Áudios Redundantes - Alvo: {alvo.upper()} (Remoção: {remover_arquivos})[/bold red]")
    
    audios_removidos = 0
    total_analisado = 0

    if alvo == "local":
        # Varre a pasta de trabalho (files/)
        for root, dirs, files in os.walk(BASE_DIR):
            if "video_text.json" in files:
                caminho_pasta = Path(root)
                total_analisado += 1
                
                # Procura por qualquer arquivo .mp3 na pasta
                arquivos_audio = list(caminho_pasta.glob("*.mp3"))
                
                for audio_path in arquivos_audio:
                    acao = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]IDENTIFICADO (Teste)[/bold yellow]"
                    console.print(f"{acao} Áudio Local: {audio_path.relative_to(BASE_DIR)}")
                    
                    if remover_arquivos:
                        try:
                            audio_path.unlink()
                            audios_removidos += 1
                        except Exception as e:
                            console.print(f"   └── [red]Erro: {e}[/red]")
                    else:
                        audios_removidos += 1

    elif alvo == "remoto":
        # Varre a pasta de versionamento (data/) usando os índices Parquet como guia
        if not Path('data').exists():
            console.print("[red]Pasta 'data' não encontrada.[/red]")
            return

        for parquet_file in Path('data').glob("*_index.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                # Filtra vídeos que têm transcrição E ainda têm áudio marcado
                if 'has_transcript' in df.columns and 'has_audio' in df.columns:
                    redundantes = df[(df['has_transcript'] == True) & (df['has_audio'] == True)]
                    
                    for _, row in redundantes.iterrows():
                        total_analisado += 1
                        path_audio_str = row.get('audio_path')
                        
                        if path_audio_str and pd.notna(path_audio_str):
                            path_audio = Path(path_audio_str)
                            
                            if path_audio.exists():
                                acao = "[bold red]EXPURGANDO PAYLOAD[/bold red]" if remover_arquivos else "[bold yellow]REDUNDANTE NO PAYLOAD (Teste)[/bold yellow]"
                                console.print(f"{acao} ID: {row['video_id']} | Arquivo: {path_audio.name}")
                                
                                if remover_arquivos:
                                    try:
                                        path_audio.unlink()
                                        audios_removidos += 1
                                    except Exception as e:
                                        console.print(f"   └── [red]Erro ao deletar físico: {e}[/red]")
                                else:
                                    audios_removidos += 1
            except Exception as e:
                console.print(f"[red]Erro ao processar índice {parquet_file.name}: {e}[/red]")

    console.print(f"\n[bold]Resumo da Faxina ({alvo}):[/bold]")
    console.print(f"Vídeos com transcrição validados: {total_analisado}")
    console.print(f"Áudios removidos/identificados: {audios_removidos}")

if __name__ == "__main__":
    # limpar_pastas_duplicadas(remover_arquivos=False)

    # gerenciar_transcricoes_erradas(remover_arquivos=False)

    # padronizar_nomes_pastas(remover_arquivos=False)
    
    # filtrar_por_data(
    #     data_inicio="2020-01-01", 
    #     data_fim="2023-01-31", 
    #     remover_arquivos=False
    # )

    #sincronizar_datas_dashboard()

    # filtrar_por_palavras_chave(remover_arquivos=False)

    expurgar_audios_redundantes(alvo='local', remover_arquivos=False)