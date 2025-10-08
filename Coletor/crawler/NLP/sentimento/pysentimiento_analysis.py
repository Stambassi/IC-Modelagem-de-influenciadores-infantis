import os
import json
import pandas as pd
from pathlib import Path
from rich.console import Console

from pysentimiento import create_analyzer

console = Console()

analyzer = create_analyzer(task="sentiment", lang="pt")

'''
    Função para analisar a toxicidade de determinado texto
    @param texto - Lista com o trecho a ser analisado pelo modelo
    @return Dict - Dicionário com as probabilidades NEG, NEU e POS
'''
def analisar_sentimento(texto: str) -> dict:
    return analyzer.predict(texto).probas

'''
    Função para percorrer as pastas dos youtubers, encontrar os vídeos com transcrição e aplicar a análise de sentimento em cada uma das tiras
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def atualizar_tiras_sentimento(youtubers_list: list[str]) -> None:
    # Percorrer a lista de nomes de youtubers
    for youtuber in youtubers_list:
        console.print(f"[bold blue]>>>>>> Processando YouTuber: {youtuber}[/bold blue]")
        
        base_path = Path(f"files/{youtuber}")
        
        if not base_path.is_dir():
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado em '{base_path}'. Pulando.[/yellow]")
            continue

        # Buscar pelo arquivo de tiras de um vídeo
        for tiras_path in base_path.rglob('tiras_video.csv'):
            console.print(f"  -> Processando arquivo: {tiras_path}")
            
            try:
                # Carregar o CSV
                data = pd.read_csv(tiras_path)

                # Verificar se o arquivo já foi processado para evitar retrabalho
                if 'negatividade' in data.columns:
                    console.print(f"     [yellow]Aviso: Arquivo já contém a coluna 'negatividade' (sentimento). Pulando para evitar reprocessamento.[/yellow]")
                    continue
                
                # Garantir que a coluna 'tiras' existe e não está vazia
                if 'tiras' not in data or data['tiras'].dropna().empty:
                    console.print(f"     [yellow]Aviso: Arquivo não contém dados na coluna 'tiras'. Pulando.[/yellow]")
                    continue

                # Aplicar a função de análise a cada tira de uma vez.
                resultados_series = data['tiras'].dropna().apply(analisar_sentimento) # Resultado é uma série de dicionários

                # Converter a Série de dicionários em um novo DataFrame
                resultados_df = pd.DataFrame(resultados_series.tolist(), index=resultados_series.index)

                # Encontrar o sentimento dominante de forma vetorizada (muito rápido)
                resultados_df['negatividade'] = resultados_df.idxmax(axis=1)

                # Renomear as colunas
                resultados_df.rename(columns={'POS': 'positividade', 'NEU': 'neutralidade', 'NEG': 'negatividade'}, inplace=True)
                
                # Juntar o DataFrame original com os resultados da análise
                data_atualizado = data.join(resultados_df)
                
                # Salvar o arquivo CSV, sobrescrevendo o original com os novos dados
                data_atualizado.to_csv(tiras_path, index=False, encoding='utf-8')
                
                console.print(f"     [green]Análise de sentimento por tira concluída e salva em {tiras_path}[/green]")

            except Exception as e:
                console.print(f"     [red]Ocorreu um erro inesperado ao processar {tiras_path}: {e}[/red]")

if __name__ == '__main__' :
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    atualizar_tiras_sentimento(lista_youtubers)