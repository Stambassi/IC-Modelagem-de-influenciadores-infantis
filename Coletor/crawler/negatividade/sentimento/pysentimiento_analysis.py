import os
import json
import pandas as pd
from pathlib import Path
from rich.console import Console

from pysentimiento import create_analyzer

console = Console()

analyzer = create_analyzer(task="sentiment", lang="pt")

'''
    Função para atualizar os valores de toxicidade dos vídeos de um youtuber
    @param sentiment_dict - Dicionário com os novos valores de toxicidade da análise de sentimento
    @param csv_file_path - Arquivo csv do youtuber a ser atualizado
    @param video_name - Nome do vídeo analisado
'''
def update_data(sentiment_dict: dict, csv_file_path: str, video_name: str) -> None:
    try:
        # Criar novo dicionário no formato do arquivo csv com os valores novos
        new_row = {
            "negative": sentiment_dict["NEG"],
            "positive": sentiment_dict["POS"],
            "neutral": sentiment_dict["NEU"],
            "video_name": video_name
        }
        
        # Crar DataFrame a partir de uma lista [] de dicionários
        new_df = pd.DataFrame([new_row])
        
        # Testar se o caminho do arquivo csv existe
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
        
        # Atualizar o arquivo csv com o DataFrame atualizado
        updated_df.to_csv(csv_file_path, index=False)
    except Exception as e:
        print(f"Erro: {e}")

'''
    Função para analisar a toxicidade de determinado texto
    @param texto - Lista com o trecho a ser analisado pelo modelo
    @return Dict - Dicionário com as probabilidades NEG, NEU e POS
'''
def analisar_sentimento(texto: str) -> dict:
    return analyzer.predict(texto).probas

'''
    Função para percorrer as pastas dos youtubers, encontrar os vídeos com transcrição e aplicar a análise de sentimento/toxicidade.
    @param youtubers_list - Lista de youtubers a serem analisados.
'''
def atualizar_geral_sentimento(youtubers_list: list[str]) -> None:
    # Percorrer a lista de nomes de youtubers
    for youtuber in youtubers_list:
        console.print(f"[bold blue]>>>>>> Processando YouTuber: {youtuber}[/bold blue]")
        
        base_path = Path(f"files/{youtuber}")
        
        # Verificar se o diretório base do youtuber realmente existe
        if not base_path.is_dir():
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado em '{base_path}'. Pulando.[/yellow]")
            continue

        for json_path in base_path.rglob('video_text.json'):
            console.print(f"  -> Encontrado arquivo de transcrição: {json_path}")
            
            try:
                # Abrir o arquivo com codificação explícita
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Verificação robusta se o arquivo tem conteúdo e a chave 'text' não está vazia
                if data and 'text' in data and data['text']:
                    # O nome da pasta do vídeo é o "pai" do arquivo json
                    nome_da_pasta_video = json_path.parent.name

                    # Criar o diretório de saída e definir o caminho do CSV
                    output_dir = Path(f'{base_path}/sentimento')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    csv_file_path = output_dir / "sentiment.csv"

                    # Analisar o texto extraído do JSON
                    resultado_analise = analisar_sentimento(data['text'])
                    
                    # Atualizar o CSV com os novos dados
                    update_data(resultado_analise, csv_file_path, nome_da_pasta_video)
                    console.print(f"     [green]Análise do vídeo '{nome_da_pasta_video}' atualizada em {csv_file_path}[/green]")
                else:
                    console.print(f"     [yellow]Aviso: Arquivo JSON está vazio ou não contém a chave 'text'. Pulando.[/yellow]")
            
            except json.JSONDecodeError:
                console.print(f"     [red]Erro: Falha ao ler o arquivo JSON (mal formatado): {json_path}. Pulando.[/red]")
            except Exception as e:
                console.print(f"     [red]Ocorreu um erro inesperado ao processar {json_path}: {e}[/red]")

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
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado. Pulando.[/yellow]")
            continue

        for tiras_path in base_path.rglob('tiras_video.csv'):
            console.print(f"  -> Processando arquivo: {tiras_path}")
            
            try:
                # Carregar o CSV
                data = pd.read_csv(tiras_path)

                # Verificar se o arquivo já foi processado para evitar retrabalho
                if 'grupo' in data.columns:
                    console.print(f"     [yellow]Aviso: Arquivo já contém a coluna 'grupo'. Pulando para evitar reprocessamento.[/yellow]")
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
                resultados_df['grupo'] = resultados_df.idxmax(axis=1)

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

    atualizar_geral_sentimento(lista_youtubers)

    #atualizar_tiras_sentimento(lista_youtubers)