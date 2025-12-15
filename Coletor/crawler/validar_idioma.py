import json
import os
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from rich.console import Console

# Garante resultados determinísticos (mesmo texto sempre dá o mesmo idioma)
DetectorFactory.seed = 0

console = Console()

# ---------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------
BASE_DIR = Path("files") 
# ---------------------------------------------------------

def gerenciar_transcricoes_erradas(remover_arquivos: bool = False):
    total_analisado = 0
    arquivos_ingles = []
    erros_leitura = 0
    
    acao_texto = "[bold red]DELETANDO[/bold red]" if remover_arquivos else "[bold yellow]IDENTIFICADO (Modo Teste)[/bold yellow]"
    
    console.rule(f"[bold magenta]Validando Idioma (Remoção: {remover_arquivos})[/bold magenta]")

    # Itera sobre todos os arquivos video_text.json
    for json_path in BASE_DIR.rglob("video_text.json"):
        total_analisado += 1
        
        try:
            # Abre e lê o arquivo
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                texto = data.get('text', '')
            
            # Pula textos muito curtos (deteção não confiável)
            if len(texto) < 50:
                continue

            # Detecta o idioma
            try:
                idioma_detectado = detect(texto)
            except LangDetectException:
                idioma_detectado = "unknown"

            # Lógica de Identificação (Se não for PT)
            if idioma_detectado != 'pt':
                caminho_relativo = json_path.relative_to(BASE_DIR)
                
                console.print(f"{acao_texto} [{idioma_detectado}]: {caminho_relativo}")
                arquivos_ingles.append(caminho_relativo)

                # Ação de remoção
                if remover_arquivos:
                    try:
                        json_path.unlink() # Deleta o arquivo
                        console.print(f"   └── [green]Arquivo deletado com sucesso.[/green]")
                    except Exception as e:
                        console.print(f"   └── [red]Erro ao deletar: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Erro ao processar {json_path}: {e}[/red]")
            erros_leitura += 1

    # Relatório Final
    console.rule("[bold]Resumo da Operação[/bold]")
    console.print(f"Total de arquivos verificados: {total_analisado}")
    console.print(f"Arquivos em outro idioma: {len(arquivos_ingles)}")
    
    if remover_arquivos:
        console.print(f"[bold red]Arquivos removidos:[/bold red] {len(arquivos_ingles)}")
        console.print("[green]Agora você pode rodar seu script de transcrição novamente para refazer estes vídeos forçando PT.[/green]")
    else:
        console.print(f"[bold yellow]Nenhum arquivo foi deletado.[/bold yellow] Mude remover_arquivos = True para executar a limpeza.")

if __name__ == "__main__":
    gerenciar_transcricoes_erradas(remover_arquivos=False)