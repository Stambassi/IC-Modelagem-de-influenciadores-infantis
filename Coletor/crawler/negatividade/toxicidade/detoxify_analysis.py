import pandas as pd
from detoxify import Detoxify
from pathlib import Path
from rich.console import Console
import time

console = Console()

'''
    Função para percorrer os diretórios dos youtubers, encontrar os arquivos de tiras e aplicar a análise de toxicidade, salvando em um novo arquivo
    @param youtubers_list - Lista de youtubers a serem analisados
    @param model - Modelo do detoxify para análise de toxicidade
'''
def processar_toxicidade_youtubers(youtubers_list: list, model) -> None:
    for youtuber in youtubers_list:
        console.print(f"[bold blue]>>>>>> Processando YouTuber: {youtuber}[/bold blue]")
        base_path = Path('files') / youtuber
        
        if not base_path.is_dir():
            console.print(f"[yellow]Aviso: Diretório para '{youtuber}' não encontrado. Pulando.[/yellow]")
            continue

        # Encontrar todos os arquivos de tiras
        for input_csv_path in base_path.rglob('tiras_video.csv'):
            # Definir o nome do arquivo de saída
            output_csv_path = input_csv_path.parent / f"{input_csv_path.stem}_toxicidade.csv"

            # Verificar se o arquivo já foi processado
            if output_csv_path.exists():
                console.print(f"  -> [yellow]Já processado, pulando:[/yellow] {input_csv_path.parent.name}")
                continue

            console.print(f"  -> [bold]Analisando:[/bold] {input_csv_path}")

            try:
                df_tiras = pd.read_csv(input_csv_path)
                textos_para_analise = df_tiras['tiras'].dropna().astype(str).tolist()

                if not textos_para_analise:
                    console.print("     [yellow]Arquivo não contém texto para análise.[/yellow]")
                    continue

                # Realizar a predição em lote
                start_time = time.time()
                resultados = model.predict(textos_para_analise)
                end_time = time.time()
                
                console.print(f"     Análise de {len(textos_para_analise)} tiras concluída em {end_time - start_time:.2f} segundos.")

                df_resultados_toxicidade = pd.DataFrame(resultados)
                
                # Juntar os resultados com os dados originais
                df_final = pd.concat([df_tiras, df_resultados_toxicidade], axis=1)
                
                # Salvar o novo arquivo CSV
                df_final.to_csv(output_csv_path, index=False, encoding='utf-8')
                console.print(f"     [green]Resultados salvos em {output_csv_path}[/green]")

            except Exception as e:
                console.print(f"     [red]Ocorreu um erro inesperado ao processar {input_csv_path}: {e}[/red]")

if __name__ == '__main__':
    lista_youtubers = ['Julia MineGirl', 'Tex HS']

    console.print("[bold]Carregando o modelo Detoxify (multilingual)...[/bold]")
    console.print("[yellow]Isso pode demorar alguns minutos na primeira execução, pois o modelo será baixado (~500MB).[/yellow]")
    
    # Carregar o modelo uma única vez
    try:
        detoxify_model = Detoxify('multilingual')
        processar_toxicidade_youtubers(lista_youtubers, detoxify_model)
    except Exception as e:
        console.print(f"\n[bold red]Falha ao carregar o modelo Detoxify ou ao executar a análise. Erro: {e}[/bold red]")
        console.print("[yellow]Verifique sua conexão com a internet ou se há algum problema com a instalação do PyTorch/TensorFlow.[/yellow]")