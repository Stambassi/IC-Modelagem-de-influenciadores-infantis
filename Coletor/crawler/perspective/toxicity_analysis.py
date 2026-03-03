from googleapiclient import discovery
import json
import pandas as pd
from detoxify import Detoxify
from pathlib import Path
from rich.console import Console
import time
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from googleapiclient.errors import HttpError

API_KEY = ""

def perspective_toxicity(text):

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['summaryScore']['value']
    

console = Console()

'''
    Função para percorrer os diretórios dos youtubers, encontrar os arquivos de tiras e aplicar a análise de toxicidade, atualizando o mesmo arquivo com os novos dados
    @param youtubers_list - Lista de youtubers a serem analisados
    @param model - Modelo do detoxify para análise de toxicidade
'''
def _processar_tiras_toxicidade(youtubers_list: list) -> None:
    numero_tiras = 0

    for youtuber in youtubers_list:
        console.print(f"[bold blue]>>>>>> Processando YouTuber: {youtuber}[/bold blue]")
        base_path = Path('files') / youtuber
        
        if not base_path.is_dir():
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado. Pulando.[/yellow]")
            continue

        # Encontrar todos os arquivos de tiras
        for input_csv_path in base_path.rglob('tiras_video.csv'):
            try:
                # Carrega o arquivo CSV original
                df_tiras = pd.read_csv(input_csv_path)

                if 'p_toxicity' in df_tiras.columns:
                    console.print(f"[yellow]Arquivo já contém colunas de toxicidade com Perspective API. Pulando.[/yellow]")
                    # print(input_csv_path)
                    continue

                # Extrai os textos para análise, garantindo que não sejam nulos
                textos_para_analise = df_tiras['tiras'].dropna().astype(str).tolist()


                if not textos_para_analise:
                    # console.print("     [yellow]Arquivo não contém texto para análise.[/yellow]")
                    continue

                resultados = []

                with alive_bar(len(df_tiras.index), bar="classic2", receipt=False, title="    >> Calculando Toxicidade") as bar:
                    for texto in textos_para_analise:
                        if numero_tiras >= 60:
                            console.print("[red] !! Limite da API alcançado. Esperando 60 segundos !![/]")
                            time.sleep(60)
                            resultados.append(perspective_toxicity(texto))
                            numero_tiras = 1
                        else:
                            resultados.append(perspective_toxicity(texto))
                            numero_tiras += 1
                        bar()
                            

                #console.print(f"     Análise de {len(textos_para_analise)} tiras concluída em {end_time - start_time:.2f} segundos.")

                # Converte o dicionário de resultados em um DataFrame
                df_resultados_toxicidade = pd.DataFrame({'p_toxicity': resultados})
                
                # Juntar os resultados com os dados originais
                df_final = pd.concat([df_tiras, df_resultados_toxicidade], axis=1)
                
                # Salvar o DataFrame enriquecido de volta no arquivo original
                df_final.to_csv(input_csv_path, index=False, encoding='utf-8')
                console.print(f"    [green]>> Colunas de toxicidade adicionadas e salvas em {input_csv_path}[/green]")

            except HttpError as e:
                if e.resp.status == 429:
                    console.print("[red]    >> ERRO: Limite da API alcançado. [/]Esperando 60 segundos...")
                    time.sleep(60)
            except Exception as e:
                console.print(f"    [red]>> Ocorreu um erro inesperado ao processar {input_csv_path}: {e}[/red]")

'''
    Função principal (pública) para carregar o modelo e iniciar a análise de toxicidade.
    Esta é a função que deve ser importada por outros scripts.
    @param youtubers_list - Lista de youtubers a serem analisados
'''
def rodar_analise_toxicidade(youtubers_list: list[str]) -> None:
    try:
        _processar_tiras_toxicidade(youtubers_list)
    except Exception as e:
        console.print("[yellow]Verifique sua conexão com a internet ou se há algum problema com a instalação do PyTorch/TensorFlow.[/yellow]")
        print(e)



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
    # lista_youtubers = ['Julia MineGirl']

    try:
        with open("perspective/api_key.txt", "r") as file:
            API_KEY = file.read()
        console.print("[bold green]Executando Análise de Toxicidade (Perspective API)[/bold green]")
        rodar_analise_toxicidade(lista_youtubers)   
    except FileNotFoundError:
        print("Crie um arquivo api_key.txt e coloque a chave")


    # grafico_comparativo(pd.read_csv('files/Julia MineGirl/2023/Maio/JOGO SATISFATÓRIO de FRUTAS NO ROBLOX! (Fruit Connect)/tiras_video.csv'))

