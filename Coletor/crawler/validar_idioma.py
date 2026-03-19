import json
import os
import shutil
import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from rich.console import Console

# Garante resultados determinísticos
DetectorFactory.seed = 0
console = Console()

# Configuração
BASE_DIR = Path("files") 

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

if __name__ == "__main__":
    limpar_pastas_duplicadas(remover_arquivos=True)
    gerenciar_transcricoes_erradas(remover_arquivos=True)