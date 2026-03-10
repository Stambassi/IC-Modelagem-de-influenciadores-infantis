import pandas as pd
from detoxify import Detoxify
from pathlib import Path
from rich.console import Console
import time

console = Console()

'''
    Função para percorrer os diretórios dos youtubers, encontrar os arquivos de tiras e aplicar a análise de toxicidade, 
    atualizando o mesmo arquivo com os novos dados

    @param youtubers_list - Lista de youtubers a serem analisados
    @param model - Modelo do detoxify para análise de toxicidade
    @param nome_arquivo - Nome específico do arquivo CSV (ex: 'tiras_video_30.csv'). Se None, busca 'tiras_video.csv'.
'''
def _processar_tiras_toxicidade(youtubers_list: list, model, nome_arquivo: str = None) -> None:
    # Define o padrão de busca: se um nome_arquivo for dado, busca na pasta /tiras/
    padrao_busca = f'tiras/{nome_arquivo}' if nome_arquivo else 'tiras_video.csv'
    
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

                # Checa se a coluna 'toxicity' já existe para evitar reprocessamento dispendioso
                if 'toxicity' in df_tiras.columns:
                    continue

                # Extrai os textos para análise, garantindo integridade dos dados
                textos_para_analise = df_tiras['tiras'].dropna().astype(str).tolist()

                if not textos_para_analise:
                    continue

                # Realizar a predição em lote (batch prediction)
                resultados = model.predict(textos_para_analise)
                
                # Converte o dicionário de resultados do Detoxify em um DataFrame
                df_resultados_toxicidade = pd.DataFrame(resultados)
                
                # Juntar os resultados com os dados originais (concatenação lateral)
                df_final = pd.concat([df_tiras, df_resultados_toxicidade], axis=1)
                
                # Salvar o DataFrame enriquecido, mantendo a codificação UTF-8
                df_final.to_csv(input_csv_path, index=False, encoding='utf-8')
                console.print(f"     [green]Análise concluída e salva em {input_csv_path.name}[/green]")

            except Exception as e:
                console.print(f"     [red]Ocorreu um erro inesperado ao processar {input_csv_path}: {e}[/red]")

'''
    Função principal (pública) para carregar o modelo e iniciar a análise de toxicidade
    
    @param youtubers_list - Lista de youtubers a serem analisados
    @param nome_arquivo - Nome do arquivo CSV de granularidade específica
'''
def rodar_analise_toxicidade(youtubers_list: list[str], nome_arquivo: str = None) -> None:
    console.print("[bold]Carregando o modelo Detoxify (multilingual)...[/bold]")
    
    # Carregar o modelo uma única vez para otimização de memória e tempo
    try:
        detoxify_model = Detoxify('multilingual')

        # Chamar a função interna de processamento com o parâmetro de granularidade
        _processar_tiras_toxicidade(youtubers_list, detoxify_model, nome_arquivo)
    except Exception as e:
        console.print(f"\n[bold red]Falha ao carregar o modelo Detoxify ou ao executar a análise. Erro: {e}[/bold red]")

if __name__ == '__main__':
    # Lista de youtubers a serem analisados
    lista_youtubers = ['Amy Scarlet', 'AuthenticGames', 'Cadres', 'Julia MineGirl', 'Kass e KR', 'Lokis', 'Luluca Games', 'Papile', 'Robin Hood Gamer', 'TazerCraft', 'Tex HS']

    console.print("[bold green]Executando Análise de Toxicidade (Standalone)[/bold green]")
    
    rodar_analise_toxicidade(lista_youtubers)