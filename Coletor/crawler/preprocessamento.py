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
    PROJECT_ROOT = CURRENT_FILE_PATH.parent

    # Define os caminhos para as novas pastas (reformuladas)
    PATH_FOLDER_PYSENTIMIENTO = PROJECT_ROOT / "NLP" / "pysentimiento"
    PATH_FOLDER_DETOXIFY = PROJECT_ROOT / "NLP" / "detoxify"
    PATH_FOLDER_PERSPECTIVE = PROJECT_ROOT / "NLP" / "perspective"

    # Adiciona as pastas ao sys.path
    sys.path.append(str(PATH_FOLDER_PYSENTIMIENTO))
    sys.path.append(str(PATH_FOLDER_DETOXIFY))
    sys.path.append(str(PATH_FOLDER_PERSPECTIVE))

    # Importar os scripts das novas pastas
    from pysentimiento_analysis import atualizar_tiras_sentimento
    from detoxify_analysis import rodar_analise_toxicidade as rodar_detoxify
    from perspective_analysis import rodar_analise_toxicidade as rodar_perspective

except ImportError as e:
    console.print(f"[bold red]ERRO DE IMPORTAÇÃO:[/bold red] Não foi possível encontrar os scripts de análise.")
    console.print(f"Verifique se a estrutura de pastas está correta e se os arquivos '__init__.py' existem.")
    console.print(f"Caminho do Projeto Raiz (calculado): {PROJECT_ROOT}")
    console.print(f"Tentando carregar de: {PATH_FOLDER_PYSENTIMIENTO}")
    console.print(f"Tentando carregar de: {PATH_FOLDER_DETOXIFY}")
    console.print(f"Tentando carregar de: {PATH_FOLDER_PERSPECTIVE}")
    console.print(f"Erro detalhado: {e}")
    sys.exit(1) # Encerra o script se os módulos não puderem ser carregados
except NameError:
    # Fallback caso __file__ não esteja definido (ex: rodando em um notebook interativo)
    console.print("[yellow]Aviso: __file__ não definido. Assumindo que os scripts estão no mesmo diretório.[/yellow]")

    # Tenta importar diretamente
    from pysentimiento_analysis import atualizar_tiras_sentimento
    from detoxify_analysis import rodar_analise_toxicidade as rodar_detoxify
    from perspective_analysis import rodar_analise_toxicidade as rodar_perspective

# Definir configurações globais
BASE_DIR = Path("files")

# Mapeia cada youtuber para sua categoria principal (Minecraft, Roblox, etc.)
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
        
        # Se o tempo for -1, retorna todo o texto como uma única tira
        if tempo_alvo == -1:
            texto_completo = " ".join([s.get('text', '').strip() for s in data["segments"]])
            return [texto_completo] if texto_completo.strip() else []

        # Lógica de segmentação normal para valores positivos
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
    except TypeError:
        console.print(f"[red]Erro: Falha ao iterar sobre o JSON[/red]")
        return []        
    except Exception as e:
        console.print(f"[red]Erro inesperado em gerar_tira_frase_tempo ({data_path.name}): {e}[/red]")
        return []

'''
    Função para percorrer toda a estrutura de pastas dos arquivos coletados,
    ler os arquivos 'video_text.json', gerar as tiras de 60,
    e salvá-las em um arquivo 'tiras_video.csv' na mesma pasta
'''
def salvar_tiras_monogranular():
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
    Função orquestradora que executa o pipeline completo com tiras de tamanho padrão
    
    @param youtubers_list - Lista de youtubers a serem processados
    @param rodar_pysentimiento - Booleano para executar análise de sentimento
    @param rodar_detoxify - Booleano para executar análise Detoxify
    @param rodar_perspective - Booleano para executar análise Perspective API
'''
def executar_pipeline_processamento_monogranular(
    youtubers_list: list[str], 
    rodar_pysent: bool = True,
    rodar_detox: bool = True,
    rodar_persp: bool = True
):
    console.print("[bold magenta]===== INICIANDO PIPELINE MONO-GRANULARIDADE =====[/bold magenta]")
    
    # Gerar os arquivos CSV para cada tempo definido
    salvar_tiras_monogranular()
    
    # Análise de sentimento (Pysentimiento)
    if rodar_pysent:
        console.print(f"\n[bold green]ETAPA 2: Analisando Sentimento (Pysentimiento)...[/bold green]")
        try:
            atualizar_tiras_sentimento(youtubers_list)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 2: {e}[/bold red]")
    
    # Análise de toxicidade (Detoxify)
    if rodar_detox:
        console.print(f"\n[bold green]ETAPA 3: Analisando Toxicidade (Detoxify)...[/bold green]")
        try:
            rodar_detoxify(youtubers_list)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 3: {e}[/bold red]")

    # Análise de toxicidade (Perspective API)
    if rodar_persp:
        console.print(f"\n[bold green]ETAPA 4: Analisando Toxicidade (Perspective API)...[/bold green]")
        try:
            rodar_perspective(youtubers_list)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 4: {e}[/bold red]")
    
    console.print("\n[bold magenta]===== PIPELINE FINALIZADO PARA MONO-GRANULARIDADE =====[/bold magenta]")

'''
    Função para percorrer a estrutura de pastas, gerar tiras com multi-granularidade
    e salvá-las em uma subpasta 'tiras' dentro de cada diretório de vídeo

    @param lista_granularidade - Lista com os tempos em segundos para as janelas
'''
def salvar_tiras_multigranular(lista_granularidade: List[int] = [30, 60, 120]):
    base_dir = Path("files")
    arquivo_origem = 'video_text.json'

    console.print(f"[bold green]Iniciando geração de multi-granularidade {lista_granularidade}s...[/bold green]")
    
    arquivos_encontrados = 0
    tiras_geradas = 0

    # rglob para encontrar todos os arquivos de transcrição
    for json_path in base_dir.rglob(arquivo_origem):
        arquivos_encontrados += 1
        video_folder = json_path.parent
        
        # Cria a subpasta 'tiras' para organizar as diferentes versões
        pasta_destino = video_folder / "tiras"
        pasta_destino.mkdir(parents=True, exist_ok=True)

        for tempo in lista_granularidade:
            # Nomeia como 'global' se o tempo for o sentinela -1
            suffix = "global" if tempo == -1 else str(tempo)
            output_csv_path = pasta_destino / f"tiras_video_{suffix}.csv"

            # Pula se o arquivo desta granularidade específica já existir
            if output_csv_path.exists():
                continue
                
            try:
                tiras = gerar_tira_frase_tempo(tempo, json_path)

                if tiras:
                    df_tiras = pd.DataFrame(tiras, columns=['tiras'])  
                    df_tiras.to_csv(output_csv_path, index_label='index')
                    tiras_geradas += 1

            except Exception as e:
                console.print(f"     [red]Erro ao processar {json_path.name} ({tempo}s): {e}[/red]")

    console.print(f"     [cyan]Processo de segmentação concluído. {tiras_geradas} novos arquivos CSV criados.[/cyan]")

'''
    Função orquestradora que executa o pipeline completo adaptado para multi-granularidade
    Garante que os modelos de NLP processem todos os arquivos na pasta 'tiras'
    
    @param youtubers_list - Lista de youtubers a serem processados
    @param lista_granularidade - Janelas temporais para análise
    @param rodar_pysentimiento - Booleano para executar análise de sentimento
    @param rodar_detoxify - Booleano para executar análise Detoxify
    @param rodar_perspective - Booleano para executar análise Perspective API
'''
def executar_pipeline_processamento_multigranular(
    youtubers_list: list[str], 
    lista_granularidade: List[int] = [30, 60, 120],
    rodar_pysent: bool = True,
    rodar_detox: bool = True,
    rodar_persp: bool = True
):
    console.print("[bold magenta]===== INICIANDO PIPELINE MULTI-GRANULARIDADE =====[/bold magenta]")
    
    # Gerar os arquivos CSV para cada tempo definido
    salvar_tiras_multigranular(lista_granularidade)
    
    # Lista de sufixos de arquivos para os modelos processarem
    arquivos_alvo = [f"tiras_video_{'global' if t == -1 else t}.csv" for t in lista_granularidade]
    
    # Análise de sentimento (Pysentimiento)
    if rodar_pysent:
        console.print(f"\n[bold green]ETAPA 2: Analisando Sentimento (Pysentimiento)...[/bold green]")
        try:
            for nome_csv in arquivos_alvo:
                console.print(f"  -> Processando: {nome_csv}", style="dim")
                atualizar_tiras_sentimento(youtubers_list, nome_arquivo=nome_csv)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 2: {e}[/bold red]")
    
    # Análise de toxicidade (Detoxify)
    if rodar_detox:
        console.print(f"\n[bold green]ETAPA 3: Analisando Toxicidade (Detoxify)...[/bold green]")
        try:
            for nome_csv in arquivos_alvo:
                console.print(f"  -> Processando: {nome_csv}", style="dim")
                rodar_detoxify(youtubers_list, nome_arquivo=nome_csv)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 3: {e}[/bold red]")

    # Análise de toxicidade (Perspective API)
    if rodar_persp:
        console.print(f"\n[bold green]ETAPA 4: Analisando Toxicidade (Perspective API)...[/bold green]")
        try:
            for nome_csv in arquivos_alvo:
                console.print(f"  -> Processando: {nome_csv}", style="dim")

                rodar_perspective(youtubers_list, nome_arquivo=nome_csv)
        except Exception as e:
            console.print(f"[bold red]Erro na Etapa 4: {e}[/bold red]")
    
    console.print("\n[bold magenta]===== PIPELINE FINALIZADO PARA TODAS AS ESCALAS SELECIONADAS =====[/bold magenta]")

if __name__ == "__main__":
    lista_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    granularidade_padrao = [30, 60, 120, 180, 240, 300, -1]

    console.print("[bold cyan]Opções de Análise NLP:[/bold cyan]")
    console.print("1. Executar TODAS as análises")
    console.print("2. Apenas Pysentimiento")
    console.print("3. Apenas Detoxify")
    console.print("4. Apenas Perspective API")
    console.print("5. Apenas toxicidade (Detoxify + Perspective)")
    
    escolha = input("\nSelecione uma opção (1-5): ")

    # Configuração de flags baseada na escolha
    config = {
        "1": (True, True, True),
        "2": (True, False, False),
        "3": (False, True, False),
        "4": (False, False, True),
        "5": (False, True, True)
    }

    if escolha in config:
        pysentimiento, detoxify, perspective = config[escolha]

        salvar_tiras_monogranular()

        executar_pipeline_processamento_monogranular(
            lista_youtubers,
            rodar_pysent=pysentimiento,
            rodar_detox=detoxify,
            rodar_persp=perspective
        )

        # executar_pipeline_processamento_multigranular(
        #     lista_youtubers, 
        #     granularidade_padrao,
        #     rodar_pysent=pysentimiento,
        #     rodar_detox=detoxify,
        #     rodar_persp=perspective
        # )
    else:
        console.print("[red]Opção inválida. Abortando.[/red]")