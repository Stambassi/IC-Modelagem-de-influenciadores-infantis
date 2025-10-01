import pandas as pd
from rich.console import Console
import liwc
from collections import Counter, defaultdict
import re

console = Console()

'''
    Função para gerar tokens (palavras) a partir de um texto
    @param text - Texto a ser tokenizado
'''
def tokenize(text: str):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

'''
    Função para realizar a análise LIWC em todas as 'tiras' de um arquivo CSV de vídeo
    @param video_path - Caminho para o arquivo 'tiras_video.csv' de um vídeo
'''
def liwc_test(video_path: str) -> None:
    console.print(f"Iniciando análise LIWC para: [bold cyan]{video_path}[/bold cyan]")
    
    try:
        dic_path = 'liwc_analysis/LIWC2007_Portugues_win.dic'

        parse, category_names = liwc.load_token_parser(dic_path)

        console.print("Dicionário LIWC e função de análise carregados com sucesso.")
    except Exception as e:
        console.print(f"[bold red]Erro fatal ao carregar o dicionário LIWC: {e}[/bold red]")
        return

    try:
        df_tiras = pd.read_csv(video_path)
        texto_completo = ' '.join(df_tiras['tiras'].dropna().astype(str))

        if not texto_completo.strip():
            console.print("[yellow]Aviso: O arquivo não contém texto para análise.[/yellow]")
            return

        texto_limpo = texto_completo.lower()
        tokens_video = list(tokenize(texto_limpo))

        # Usa-se um defaultdict para agrupar as palavras por categoria
        palavras_por_categoria = defaultdict(list)
        
        for token in tokens_video:
            for categoria in parse(token):
                palavras_por_categoria[categoria].append(token)
        
        # A contagem agregada ainda pode ser gerada a partir deste dicionário se necessário
        contagem_agregada = {categoria: len(palavras) for categoria, palavras in palavras_por_categoria.items()}

        # Mostrar os resultados
        console.print("\n--- Resultados da Análise LIWC para o Vídeo ---")
        console.print(f"[bold]Contagem de Categorias:[/bold]")
        console.print(contagem_agregada)
        
        console.print("\n--- [bold]Amostra de Palavras por Categoria de Interesse[/bold] ---")
        
        # Exibir algumas categorias importantes da análise
        categorias_interesse = ['posemo', 'negemo', 'swear', 'cogmech', 'you', 'time', 'sexual']
        for cat in categorias_interesse:
            if cat in palavras_por_categoria:
                # Usa-se set() para ver apenas as palavras únicas
                palavras_unicas = sorted(list(set(palavras_por_categoria[cat])))
                console.print(f"\n[bold green]Categoria: {cat}[/bold green] ({len(palavras_por_categoria[cat])} ocorrências)")
                console.print(palavras_unicas)

    except FileNotFoundError:
        console.print(f"[bold red]Erro: Arquivo não encontrado em '{video_path}'[/bold red]")
    except Exception as e:
        console.print(f'[bold red]Erro inesperado durante a análise (liwc_test): {e}[/bold red]')

if __name__ == '__main__':
    video_path = 'files/Tex HS/2020/Abril/NAO ENTRE NA PORTA 806 (Roblox The Lost Episodes)/tiras_video.csv'
    
    liwc_test(video_path)