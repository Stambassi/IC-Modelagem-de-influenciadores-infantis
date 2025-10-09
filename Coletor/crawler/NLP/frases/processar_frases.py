import pandas as pd
from pathlib import Path
from rich.console import Console
import spacy
from detoxify import Detoxify
import time

console = Console()

YOUTUBERS_LIST = ['Julia MineGirl', 'Tex HS', 'Robin Hood Gamer']
BASE_FOLDER = Path('files')
INPUT_FILENAME = 'tiras_video.csv'
OUTPUT_FILENAME = 'analise_por_frase.csv'

'''
    Função para ler os arquivos de tiras, segmentar em frases, analisar a toxicidade de cada frase e salvar em um novo arquivo (para cada youtuber)
    @param youtubers_list - Lista de youtubers a serem analisados
    @param nlp_model - Modelo de NLP para segmentação do texto em frases
    @param detoxify_model - Modelo de toxicidade
'''
def processar_frases_youtubers(youtubers_list: list, nlp_model, detoxify_model):
    # Percorrer cada youtuber
    for youtuber in youtubers_list:
        console.print(f"\n[bold blue]>>>> Processando frases para: {youtuber}[/bold blue]")
        youtuber_path = BASE_FOLDER / youtuber
        
        if not youtuber_path.is_dir():
            continue

        # Buscar os arquivos de tiras de um vídeo
        for input_csv_path in youtuber_path.rglob(INPUT_FILENAME):
            # Definir o nome de pastas e arquivos
            video_folder = input_csv_path.parent
            video_id = pd.read_csv(f'{video_folder}/videos_info.csv')['video_id'][0]
            output_csv_path = video_folder / OUTPUT_FILENAME

            # Testar se o arquivo de saída já existe
            if output_csv_path.exists():
                console.print(f"  -> [yellow]Frases já analisadas, pulando:[/yellow] {video_folder.name}")
                continue
            
            console.print(f"  -> Segmentando e analisando frases em: [cyan]{video_folder.name}[/cyan]")

            try:
                # Ler o arquivo de tiras de um vídeo
                df_tiras = pd.read_csv(input_csv_path)
                
                # Testar se o arquivo é válido
                if df_tiras.empty or 'tiras' not in df_tiras.columns:
                    continue

                # Coletar todas as frases de todas as tiras
                sentencas_info = []

                # Iterar sobre todas as linhas do DataFrame de tiras
                for id_tira, row in df_tiras.iterrows():
                    # Separar o texto da tira
                    tira_text = str(row['tiras'])

                    # Testar se a tira identificada é válida
                    if not tira_text.strip():
                        continue
                    
                    # Separar o texto em frases
                    doc = nlp_model(tira_text)

                    # Iterar sobre todas as frases identificadas pelo modelo de NLP
                    for id_frase, sent in enumerate(doc.sents):
                        # Processar o texto da tira
                        frase_text = sent.text.strip()

                        # Testar se a frase é válida e, em caso afirmativo, adicionar à lista de frases
                        if frase_text:
                            sentencas_info.append({
                                'id_video': video_folder.name,
                                'id_tira': id_tira,
                                'id_frase_na_tira': id_frase,
                                'frase': frase_text
                            })
                
                # Testar se alguma frase foi encontrada
                if not sentencas_info:
                    console.print("     [yellow]Nenhuma frase encontrada no arquivo.[/yellow]")
                    continue

                # Separar apenas a lista de textos para passar ao Detoxify
                lista_de_frases = [info['frase'] for info in sentencas_info]

                # Analisar todas as frases do vídeo em um único lote
                start_time = time.time()
                resultados_toxicidade = detoxify_model.predict(lista_de_frases)
                end_time = time.time()
                console.print(f"     Análise de {len(lista_de_frases)} frases concluída em {end_time - start_time:.2f} segundos.")

                # Combinar as informações com os resultados
                df_sentencas = pd.DataFrame(sentencas_info)
                df_toxicidade = pd.DataFrame(resultados_toxicidade)
                
                df_final = pd.concat([df_sentencas, df_toxicidade], axis=1)
                
                # Salvar o novo arquivo CSV
                df_final.to_csv(output_csv_path, index=False, encoding='utf-8')
                console.print(f"     [green]Resultados por frase salvos em {output_csv_path}[/green]")

            except Exception as e:
                console.print(f"     [bold red]Erro ao processar {video_folder.name}: {e}[/bold red]")

if __name__ == '__main__':
    console.print("[bold]Carregando modelos (spaCy e Detoxify)...[/bold]")

    # Carrega o spaCy para segmentação de frases
    nlp = spacy.load('pt_core_news_lg', disable=['parser', 'ner'])
    nlp.add_pipe('sentencizer')

    # Carrega o Detoxify para análise de toxicidade
    detoxify = Detoxify('multilingual')
    
    processar_frases_youtubers(YOUTUBERS_LIST, nlp, detoxify)