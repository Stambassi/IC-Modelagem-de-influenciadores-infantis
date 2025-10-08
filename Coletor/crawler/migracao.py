from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()

YOUTUBERS_LIST = ['Robin Hood Gamer']
BASE_FOLDER = Path('files')
OLD_SENTIMENT_FILENAME = 'tiras_video.csv'
OLD_TOXICITY_FILENAME = 'tiras_video_toxicidade.csv'
NEW_UNIFIED_FILENAME = 'tiras_video.csv'
DELETAR_ARQUIVOS_ANTIGOS = True 

'''
    Função para varrer as pastas dos youtubers, unificar os arquivos de análise e adotar a nova padronização
'''
def migrar_arquivos():
    console.print("[bold green]Iniciando migração para a nova estrutura de arquivos...[/bold green]")

    for youtuber in YOUTUBERS_LIST:
        console.print(f"\n[bold blue]>>>> Processando YouTuber: {youtuber}[/bold blue]")
        youtuber_path = BASE_FOLDER / youtuber
        
        # Encontrar um dos arquivos antigos para localizar as pastas de vídeo
        for old_sent_path in youtuber_path.rglob(OLD_SENTIMENT_FILENAME):
            video_folder = old_sent_path.parent
            old_toxi_path = video_folder / OLD_TOXICITY_FILENAME
            new_unified_path = video_folder / NEW_UNIFIED_FILENAME
            
            console.print(f"  -> Migrando dados da pasta: [cyan]{video_folder.name}[/cyan]")

            if not old_toxi_path.exists():
                console.print(f"     [yellow]Aviso: Arquivo de toxicidade não encontrado. Pulando pasta.[/yellow]")
                continue
            
            try:
                # Carregar os dados de toxicidade, que já são um superset
                df_toxi = pd.read_csv(old_toxi_path)
                
                # Renomear a coluna 'grupo' para ser mais descritiva
                if 'grupo' in df_toxi.columns:
                    df_toxi.rename(columns={'grupo': 'sentimento_dominante'}, inplace=True)

                # Salvar o novo arquivo unificado
                df_toxi.to_csv(new_unified_path, index=False, encoding='utf-8')
                console.print(f"     [green]Novo arquivo unificado salvo em {new_unified_path}[/green]")

                # Apagar os arquivos antigos se a flag estiver ativa
                if DELETAR_ARQUIVOS_ANTIGOS:
                    old_sent_path.unlink() # .unlink() é o comando para apagar arquivos no pathlib
                    old_toxi_path.unlink()
                    console.print(f"     [dim]Arquivos antigos foram removidos.[/dim]")

            except Exception as e:
                console.print(f"     [bold red]Erro ao processar a pasta {video_folder.name}: {e}[/bold red]")

if __name__ == "__main__":
    migrar_arquivos()