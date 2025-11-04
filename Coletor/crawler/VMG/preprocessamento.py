# Gerar tiras a partir dos arquivos gerados pelo Whisper (video_process.py)

# Rodar a análise de sentimento e de toxicidade

# Criar os arquivos de transição para cada vídeo

import pandas as pd
from pathlib import Path
from rich.console import Console
import json
from typing import List
import sys

console = Console()

try:
    # Encontra o caminho absoluto do script atual
    CURRENT_FILE_PATH = Path(__file__).resolve()
    # Encontra a pasta pai
    PARENT_DIR = CURRENT_FILE_PATH.parent
    # Encontra a pasta raiz
    PROJECT_ROOT = PARENT_DIR.parent

    # Define os caminhos corretos para as pastas dos scripts
    PATH_FOLDER_SENTIMENTO = PROJECT_ROOT / "NLP" / "sentimento"
    PATH_FOLDER_TOXICIDADE = PROJECT_ROOT / "NLP" / "toxicidade"

    # Adiciona as pástas ao sys.path
    sys.path.append(str(PATH_FOLDER_SENTIMENTO))
    sys.path.append(str(PATH_FOLDER_TOXICIDADE))

    # Importar os scripts
    from pysentimiento_analysis import atualizar_tiras_sentimento
    from detoxify_analysis import rodar_analise_toxicidade

except ImportError as e:
    console.print(f"[bold red]ERRO DE IMPORTAÇÃO:[/bold red] Não foi possível encontrar os scripts de análise.")
    console.print(f"Verifique se a estrutura de pastas está correta e se os arquivos '__init__.py' existem.")
    console.print(f"Caminho do Projeto Raiz (calculado): {PROJECT_ROOT}")
    console.print(f"Tentando carregar de: {PATH_FOLDER_SENTIMENTO}")
    console.print(f"Tentando carregar de: {PATH_FOLDER_TOXICIDADE}")
    console.print(f"Erro detalhado: {e}")
    sys.exit(1) # Encerra o script se os módulos não puderem ser carregados
except NameError:
    # Fallback caso __file__ não esteja definido (ex: rodando em um notebook interativo)
    console.print("[yellow]Aviso: __file__ não definido. Assumindo que os scripts estão no mesmo diretório.[/yellow]")

    # Tenta importar diretamente
    from pysentimiento_analysis import atualizar_tiras_sentimento
    from detoxify_analysis import rodar_analise_toxicidade

# Definir configurações globais
YOUTUBERS_LIST = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']
BASE_DIR = Path("files")

'''
    Função para realizar o agrupamento dos segments em grupos de X segundos, mantendo a coerencia de frases
    @param tempo_alvo - Duração alvo de cada tira em segundos (ex: 60)
    @param data_path - Caminho (Path) para o arquivo json com resultado da analise do whisper
    @param margem_percent - Porcentagem de "folga" para buscar o fim da frase (padrão: 10%)
    @return List[str] - Lista de tiras de texto processadas
'''
def gerar_tira_frase_tempo(tempo_alvo: int, data_path: Path, margem_percent: int = 10) -> List[str]:    
    # Calcular o tempo mínimo antes de procurar por um corte
    try:
        tempo_real = tempo_alvo * (1 - (margem_percent / 100))
    except TypeError:
        console.print(f"[red]Erro: Parâmetros 'tempo_alvo' e 'margem_percent' devem ser numéricos.[/red]")
        return []

    try:
        # Abrir e ler o arquivo JSON de transcrição com codificação explícita
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Verificar se a estrutura esperada do JSON (chave 'segments') existe
        if "segments" not in data:
            console.print(f"[red]Erro: Arquivo JSON {data_path.name} não contém a chave 'segments'[/red]")
            return []

        tempo_acumulado = 0.0
        tira_atual = ""
        tiras_finais = []
        
        # Iterar sobre cada segmento de fala transcrito pelo Whisper
        for segment in data["segments"]:
            # Acumular o tempo e o texto do segmento atual (garante que 'text' seja uma string)
            segment_text = segment.get('text', '')
            segment_duration = segment.get('end', 0.0) - segment.get('start', 0.0)
            
            tempo_acumulado += segment_duration
            tira_atual += segment_text
            
            # Verificar se o tempo acumulado atingiu o alvo mínimo
            if (tempo_acumulado >= tempo_real):
                # Procurar o índice do último terminador de frase ('.', '?', '!')
                i = max(
                    tira_atual.rfind("."), 
                    tira_atual.rfind("?"), 
                    tira_atual.rfind("!")
                )
                
                # Se não encontrar nenhum terminador (ex: fala contínua sem pontuação)
                if i < 0: 
                    # Forçar o corte ao final da string atual
                    i = len(tira_atual)
                
                # Adicionar a parte principal (até o corte) à lista de tiras. O +1 inclui o próprio terminador (., ?, !)
                tiras_finais.append(tira_atual[:i+1].strip())
                
                # Manter o "resto" da string (após o terminador) para a próxima iteração
                tira_atual = tira_atual[i+1:]
                
                # Resetar o contador de tempo
                tempo_acumulado = 0
        
        # Após o loop, adicionar qualquer texto restante como a última tira
        if len(tira_atual.strip()) > 0:
            tiras_finais.append(tira_atual.strip())

        return tiras_finais

    except FileNotFoundError:
        console.print(f"[red]Erro: Arquivo JSON não encontrado em {data_path}[/red]")
        return [] # Retorna lista vazia em caso de erro
    except json.JSONDecodeError:
        console.print(f"[red]Erro: Falha ao decodificar o JSON em {data_path}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Erro inesperado em gerar_tira_frase_tempo ({data_path.name}): {e}[/red]")
        return []

'''
    Função para percorrer toda a estrutura de pastas dos arquivos coletados,
    ler os arquivos 'video_text.json', gerar as tiras de 1 minuto 
    e salvá-las em um arquivo 'tiras_video.csv' na mesma pasta.
'''
def salvar_tiras():
    # Definir o diretório base da busca
    base_dir = Path("files")
    
    # Definir o nome do arquivo de origem que queremos encontrar
    arquivo_origem = 'video_text.json'
    
    # Definir o nome do arquivo de destino que queremos criar
    arquivo_destino = 'tiras_video.csv'
    
    # Definir o intervalo da tira (ex: 60 segundos)
    intervalo_segundos = 60

    console.print(f"[bold green]Iniciando geração de 'tiras' a partir de '{arquivo_origem}'...[/bold green]")
    
    # Usar .rglob() para encontrar recursivamente todos os 'video_text.json'
    # dentro do diretório base. Isso substitui todos os loops aninhados.
    arquivos_encontrados = 0
    arquivos_gerados = 0

    for json_path in base_dir.rglob(arquivo_origem):
        arquivos_encontrados += 1
        video_folder = json_path.parent
        output_csv_path = video_folder / arquivo_destino

        console.print(f"  -> Processando: {json_path.relative_to(base_dir)}")

        # Verificar se o arquivo de destino JÁ EXISTE para evitar reprocessamento
        if output_csv_path.exists():
            console.print(f"     [yellow]Aviso: '{arquivo_destino}' já existe. Pulando.[/yellow]")
            continue
            
        try:
            # Calcular as tiras de cada vídeo
            # (Assumindo que 'gerar_tira_frase_tempo' existe e aceita um Path)
            tiras = gerar_tira_frase_tempo(intervalo_segundos, json_path)

            if not tiras:
                console.print(f"     [yellow]Aviso: Nenhuma tira gerada para este arquivo.[/yellow]")
                continue

            # Converter a lista para DataFrame
            df_tiras = pd.DataFrame(tiras, columns=['tiras'])  

            # Salvar a lista gerada em um arquivo .csv
            # 'index_label' é usado para nomear a coluna de índice (0, 1, 2...)
            df_tiras.to_csv(output_csv_path, index_label='index')
            console.print(f"     [green]Arquivo '{arquivo_destino}' salvo com sucesso.[/green]")
            arquivos_gerados += 1

        except Exception as e:
            console.print(f"     [red]Erro ao processar {json_path}: {e}[/red]")

    console.print(f"\n[bold green]Processo concluído![/bold green]")
    console.print(f"Arquivos '{arquivo_origem}' encontrados: {arquivos_encontrados}")
    console.print(f"Novos arquivos '{arquivo_destino}' gerados: {arquivos_gerados}")

'''
    Função orquestradora que executa o pipeline de pré-processamento completo.
    @param youtubers_list - Lista de youtubers a serem processados.
'''
def executar_pipeline_processamento(youtubers_list: list[str]):
    console.print("[bold magenta]===== INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO COMPLETO =====[/bold magenta]")
    
    # Gerar os arquivos 'tiras_video.csv'
    salvar_tiras()
    
    console.print("\n[bold magenta]==========================================================[/bold magenta]")
    
    # Rodar a análise de sentimento (Pysentimiento)
    console.print(f"[bold green]ETAPA 2: Iniciando Análise de Sentimento...[/bold green]")
    try:
        atualizar_tiras_sentimento(youtubers_list)
        console.print(f"\n[bold green]ETAPA 2 Concluída![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]FALHA NA ETAPA 2 (Sentimento): {e}[/bold red]")
    
    console.print("\n[bold magenta]==========================================================[/bold magenta]")

    # Rodar a análise de toxicidade (Detoxify)
    console.print(f"[bold green]ETAPA 3: Iniciando Análise de Toxicidade...[/bold green]")
    try:
        rodar_analise_toxicidade(youtubers_list)
        console.print(f"\n[bold green]ETAPA 3 Concluída![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]FALHA NA ETAPA 3 (Toxicidade): {e}[/bold red]")
    
    console.print("\n[bold magenta]===== PIPELINE DE PRÉ-PROCESSAMENTO FINALIZADO =====[/bold magenta]")


if __name__ == "__main__":
    executar_pipeline_processamento(YOUTUBERS_LIST)