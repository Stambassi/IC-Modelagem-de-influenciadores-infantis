from googleapiclient import discovery
import json
import pandas as pd
from pathlib import Path
from rich.console import Console
import time
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from googleapiclient.errors import HttpError

# Global para facilitar o acesso nas funções standalone, mas alimentada via rodar_analise_toxicidade
API_KEY = ""

'''
    Função para realizar a requisição de toxicidade para um texto específico utilizando o cliente da API
    @param text - Texto a ser analisado
    @param client - Cliente discovery build já inicializado
'''
def perspective_toxicity(text, client):
    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['summaryScore']['value']

console = Console()

'''
    Função para percorrer os diretórios dos youtubers, encontrar os arquivos de tiras e aplicar a análise de toxicidade, 
    atualizando o mesmo arquivo com os novos dados
    
    @param youtubers_list - Lista de youtubers a serem analisados
    @param client - Cliente discovery build da Perspective API
    @param nome_arquivo - Nome do arquivo CSV a ser buscado (ex: tiras_video.csv ou tiras_video_60.csv)
'''
def _processar_tiras_toxicidade(youtubers_list: list, client, nome_arquivo: str = None) -> None:
    # Define o padrão de busca: se um nome_arquivo for dado, busca na pasta /tiras/
    padrao_busca = f'{nome_arquivo}' if nome_arquivo else 'tiras_video.csv'

    numero_tiras = 0

    for youtuber in youtubers_list:
        console.print(f"[bold blue]>>>>>> Processando YouTuber: {youtuber}[/bold blue]")
        base_path = Path('files') / youtuber
        
        if not base_path.is_dir():
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado. Pulando.[/yellow]")
            continue
        
        # Encontrar os arquivos de tiras conforme o padrão de granularidade
        for input_csv_path in base_path.rglob(padrao_busca):
            try:
                # Carrega o arquivo CSV original
                df_tiras = pd.read_csv(input_csv_path)

                if 'p_toxicity' in df_tiras.columns:
                    console.print(f"     [yellow]Arquivo {input_csv_path.name} já contém colunas do Perspective. Pulando.[/yellow]")
                    continue

                # Extrai os textos para análise, garantindo que não sejam nulos
                textos_para_analise = df_tiras['tiras'].dropna().astype(str).tolist()

                if not textos_para_analise:
                    console.print(f"     [yellow]Não há textos para análise em {input_csv_path}[/yellow]")
                    continue

                resultados = []

                with alive_bar(len(textos_para_analise), bar="classic2", receipt=False, title=f"    >> Analisando {input_csv_path.name}") as bar:
                    for texto in textos_para_analise:
                        if numero_tiras >= 60:
                            console.print("[red] !! Limite da API alcançado. Esperando 60 segundos !![/]")
                            time.sleep(60)
                            resultados.append(perspective_toxicity(texto, client))
                            numero_tiras = 1
                        else:
                            resultados.append(perspective_toxicity(texto, client))
                            numero_tiras += 1
                        bar()
                            
                # Converte o dicionário de resultados em um DataFrame
                df_resultados_toxicidade = pd.DataFrame({'p_toxicity': resultados})
                
                # Juntar os resultados com os dados originais
                df_final = pd.concat([df_tiras, df_resultados_toxicidade], axis=1)
                
                # Salvar o DataFrame enriquecido de volta no arquivo original
                df_final.to_csv(input_csv_path, index=False, encoding='utf-8')
                console.print(f"    [green]>> Colunas de toxicidade salvas em {input_csv_path.name}[/green]")

            except HttpError as e:
                if e.resp.status == 429:
                    console.print("[red]    >> ERRO: Limite da API alcançado. [/]Esperando 60 segundos...")
                    time.sleep(60)
            except Exception as e:
                console.print(f"    [red]>> Ocorreu um erro inesperado ao processar {input_csv_path.name}: {e}[/red]")

'''
    Função principal (pública) para carregar o modelo e iniciar a análise de toxicidade.
    Esta é a função que deve ser importada por outros scripts.
    @param youtubers_list - Lista de youtubers a serem analisados
    @param nome_arquivo - Nome do arquivo CSV a ser processado (padrão: tiras_video.csv)
'''
def rodar_analise_toxicidade(youtubers_list: list[str], nome_arquivo: str = 'tiras_video.csv') -> None:
    global API_KEY
    
    # Se a API_KEY estiver vazia, tenta carregar do arquivo padrão
    if not API_KEY:
        try:
            with open("NLP/perspective/api_key.txt", "r") as file:
                API_KEY = file.read().strip()
        except FileNotFoundError:
            console.print("[bold red]Erro:[/bold red] API_KEY não definida e arquivo 'api_key.txt' não encontrado.")
            return

    try:
        # Inicializa o cliente uma única vez para todas as requisições (Otimização)
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        _processar_tiras_toxicidade(youtubers_list, client, nome_arquivo)
        
    except Exception as e:
        console.print("[yellow]Verifique sua conexão ou se há algum problema com a API Key.[/yellow]")
        console.print(f"[red]Erro detalhado: {e}[/red]")

def grafico_comparativo(df):
    plt.figure()
    
    # Plot both lines with different colors
    plt.plot(df["toxicity"].values, color="blue", label="Detoxify")
    plt.plot(df["p_toxicity"].values, color="red", label="Perspective API")
    
    plt.title("Detoxify Toxicity X Perspective Toxicity")
    plt.xlabel("Tirinha")
    plt.ylabel("Score")
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    # Lista de youtubers a serem analisados
    lista_youtubers = ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'Papile', 'Robin Hood Gamer', 'TazerCraft', 'Tex HS']

    console.print("[bold green]Executando Análise de Toxicidade (Perspective API)[/bold green]")

    rodar_analise_toxicidade(lista_youtubers)