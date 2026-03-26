import json
import os
import shutil
import pandas as pd
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from rich.console import Console
from rich.rule import Rule

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

if __name__ == "__main__":
    limpar_pastas_duplicadas(remover_arquivos=False)
    gerenciar_transcricoes_erradas(remover_arquivos=False)
    padronizar_nomes_pastas(remover_arquivos=False)