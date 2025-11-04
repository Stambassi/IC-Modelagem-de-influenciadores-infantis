
import shutil
from pathlib import Path
from rich.console import Console

console = Console()

# Defina a pasta raiz onde estão os youtubers
BASE_DATA_FOLDER = Path('files')

# Defina os youtubers que você quer processar
YOUTUBERS_LIST = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer', 'Authentic Games'] # Adicione todos os nomes aqui

# Itens para deletar DENTRO de cada pasta de VÍDEO individual
DELETE_VIDEO_DIRS = ['transicoes']
DELETE_VIDEO_FILES = ['analise_por_frase.csv']

# Itens para deletar na RAÍZ da pasta do YOUTUBER
DELETE_YOUTUBER_DIRS = ['sentimento', 'transicoes']

# Itens para criar na RAÍZ da pasta do YOUTUBER
CREATE_YOUTUBER_DIRS = ['agrupamento', 'frequencia_palavras', 'transicoes', 'VMG']

# Arquivo "âncora" que usaremos para identificar o que é uma "pasta de vídeo"
VIDEO_ANCHOR_FILE = 'tiras_video.csv'

'''
    Função para executar a limpeza e preparação das pastas do projeto de análise
'''
def limpar_e_preparar_pastas():
    console.print(f"[bold green]Iniciando limpeza e preparação do terreno...[/bold green]")
    
    for youtuber in YOUTUBERS_LIST:
        youtuber_path = BASE_DATA_FOLDER / youtuber
        console.print(f"\n[bold blue]>>>> Processando YouTuber: {youtuber}[/bold blue]")

        if not youtuber_path.is_dir():
            console.print(f"  [yellow]Aviso: Diretório não encontrado. Pulando: {youtuber_path}[/yellow]")
            continue

        # Limpeza no nível do youtuber
        console.print("  Limpando diretórios na raiz do Youtuber...")
        for dir_name in DELETE_YOUTUBER_DIRS:
            dir_to_delete = youtuber_path / dir_name
            if dir_to_delete.is_dir():
                try:
                    # shutil.rmtree é usado para deletar pastas com conteúdo
                    shutil.rmtree(dir_to_delete)
                    console.print(f"    [red]DELETADO (DIR):[/red] {dir_to_delete.relative_to(BASE_DATA_FOLDER)}")
                except Exception as e:
                    console.print(f"    [red]ERRO ao deletar {dir_to_delete}: {e}[/red]")
            else:
                console.print(f"    [dim]Não encontrado (OK): {dir_to_delete.relative_to(BASE_DATA_FOLDER)}[/dim]")

        # Limpeza no nível dos vídeos
        console.print(f"  Buscando pastas de vídeo (via '{VIDEO_ANCHOR_FILE}') para limpeza interna...")
        found_videos = 0
        
        # Usamos o .rglob para encontrar todos os arquivos âncora e, assim, as pastas de vídeo
        for anchor_file in youtuber_path.rglob(VIDEO_ANCHOR_FILE):
            video_folder = anchor_file.parent
            found_videos += 1
            
            # Deletar pastas de vídeo
            for dir_name in DELETE_VIDEO_DIRS:
                dir_to_delete = video_folder / dir_name
                if dir_to_delete.is_dir():
                    try:
                        shutil.rmtree(dir_to_delete)
                        console.print(f"    [red]DELETADO (SUB-DIR):[/red] {dir_to_delete.relative_to(BASE_DATA_FOLDER)}")
                    except Exception as e:
                        console.print(f"    [red]ERRO ao deletar {dir_to_delete}: {e}[/red]")
            
            # Deletar arquivos de vídeo
            for file_name in DELETE_VIDEO_FILES:
                file_to_delete = video_folder / file_name
                if file_to_delete.is_file():
                    try:
                        file_to_delete.unlink() # .unlink() é o comando para deletar arquivos
                        console.print(f"    [red]DELETADO (FILE):[/red] {file_to_delete.relative_to(BASE_DATA_FOLDER)}")
                    except Exception as e:
                        console.print(f"    [red]ERRO ao deletar {file_to_delete}: {e}[/red]")
        
        if found_videos == 0:
             console.print(f"    [yellow]Nenhum vídeo (com {VIDEO_ANCHOR_FILE}) encontrado para limpeza interna.[/yellow]")

        # Criar pastas de análise
        console.print("  Criando diretórios de análise no nível do Youtuber...")
        for dir_name in CREATE_YOUTUBER_DIRS:
            dir_to_create = youtuber_path / dir_name
            try:
                # .mkdir() cria a pasta, exist_ok=True não dá erro se ela já existir
                dir_to_create.mkdir(parents=True, exist_ok=True)
                console.print(f"    [green]GARANTIDO (DIR):[/green] {dir_to_create.relative_to(BASE_DATA_FOLDER)}")
            except Exception as e:
                 console.print(f"    [red]ERRO ao criar {dir_to_create}: {e}[/red]")

    console.print(f"\n[bold green]Limpeza e preparação concluídas![/bold green]")


if __name__ == "__main__":
    limpar_e_preparar_pastas()
    
    console.print("[bold yellow]Script de limpeza pronto.[/bold yellow]")