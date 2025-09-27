import pandas as pd
from pathlib import Path
import liwc
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()

'''
    Função para carregar as palavras-chave de cada youtuber a partir dos arquivos CSV
    @param youtubers_list - Lista dos youtubers a serem analisados
    @param base_folder - Pasta raiz de todos os dados
    @param analysis_folder - Nome da pasta de análise de TF-IDF
    @return keywords_por_youtuber - Palavras-chave de um youtuber
'''
def carregar_keywords(youtubers_list: list[str], base_folder: str, analysis_folder: str) -> dict:
    # Criar dicionário de palavras-chave
    keywords_por_youtuber = {}

    console.print("[bold]Passo 1: Carregando Palavras-Chave do TF-IDF[/bold]")

    for youtuber in youtubers_list:
        caminho_arquivo = Path(base_folder) / youtuber / analysis_folder / 'ranking_tfidf.csv'

        if caminho_arquivo.exists():
            df_keywords = pd.read_csv(caminho_arquivo)

            keywords_por_youtuber[youtuber] = df_keywords['palavra_chave'].tolist()

            console.print(f"Palavras-chave carregadas para [cyan]{youtuber}[/cyan].")
        else:
            console.print(f"[red]Aviso: Arquivo não encontrado para {youtuber} em {caminho_arquivo}[/red]")
            keywords_por_youtuber[youtuber] = []

    return keywords_por_youtuber

'''
    Função para analisar as palavras-chave com o LIWC e retornar um DataFrame comparativo com os resultados normalizados (em %)
    @param keywords_data - Dicionário com as palavras-chave de todos os youtubers
    @param dic_path - Caminho para o dicionário em portugês do LIWC
    @return df_normalizado - DataFrame comparativo com os resultados normalizados
'''
def analisar_com_liwc(keywords_data: dict, dic_path: str) -> pd.DataFrame:
    console.print("\n[bold]Passo 2: Analisando com LIWC e Normalizando Resultados[/bold]")
    
    try:
        parse_func, _ = liwc.load_token_parser(dic_path)
    except Exception as e:
        console.print(f"[bold red]Erro fatal ao carregar o dicionário LIWC: {e}[/bold red]")
        return None

    resultados_normalizados = {}
    
    for youtuber, keywords in keywords_data.items():
        if not keywords:
            continue
        
        # Contar as categorias LIWC para a lista de palavras-chave
        liwc_counts = Counter(categoria for token in keywords for categoria in parse_func(token))
        
        # Calcular o total de "acertos" em categorias para normalização
        total_categorias_encontradas = sum(liwc_counts.values())
        
        if total_categorias_encontradas == 0:
            console.print(f"[yellow]Aviso: Nenhuma palavra-chave de {youtuber} foi encontrada no dicionário LIWC.[/yellow]")
            continue
            
        # Normalizar os resultados (calcular a porcentagem)
        percentuais = {
            categoria: (count / total_categorias_encontradas) * 100
            for categoria, count in liwc_counts.items()
        }

        resultados_normalizados[youtuber] = percentuais

        console.print(f"Análise para [cyan]{youtuber}[/cyan] concluída.")

    # Converter o dicionário de resultados em um DataFrame do Pandas para fácil comparação
    df_comparativo = pd.DataFrame(resultados_normalizados).fillna(0).sort_index()
    
    return df_comparativo

'''
    Função para apresentar os resultados do LIWC por youtuber de forma clara
    @param df_resultados - DataFrame com os resultados da análise do LIWC
'''
def apresentar_resultados(df_resultados: pd.DataFrame):
    # Testar se o DataFrame de resultados é representativo
    if df_resultados.empty or len(df_resultados.columns) < 1:
        console.print("\n[red]Não há dados suficientes para uma análise.[/red]")
        return
        
    console.print("\n[bold green]Passo 3: Análise Comparativa do Teor do Discurso[/bold green]")
    
    # Top categorias individuais
    for youtuber in df_resultados.columns:
        #top_5 = df_resultados[youtuber].sort_values(ascending=False).head(5)
        top_5 = df_resultados[youtuber].sort_values(ascending=False)

        console.print(f"\n[bold]Top 5 Categorias para [cyan]{youtuber}[/cyan]:[/bold]")

        for categoria, percentual in top_5.items():
            console.print(f"- {categoria}: {percentual:.2f}%")
            
    # Tabela Comparativa Geral
    console.print("\n[bold magenta]Tabela Comparativa Geral (%)[/bold magenta]")
    
    tabela_geral = Table(title="Comparativo de Categorias LIWC por YouTuber")
    tabela_geral.add_column("Categoria LIWC", style="cyan", no_wrap=True)

    # Adicionar uma coluna para cada youtuber dinamicamente
    for youtuber in df_resultados.columns:
        tabela_geral.add_column(youtuber, style="magenta", justify="right")
    
    # Adicionar as linhas da tabela dinamicamente
    for categoria, row_data in df_resultados.head(15).iterrows():
        # Criar a lista de valores para a linha: [nome_categoria, valor1, valor2, ...]
        valores_linha = [categoria]
        for youtuber in df_resultados.columns:
            valores_linha.append(f"{row_data[youtuber]:.2f}%")
        
        # Adicionar a linha à tabela usando o operador splat (*)
        tabela_geral.add_row(*valores_linha)
        
    console.print(tabela_geral)

    # Encontrar características marcantes com variância
    if len(df_resultados.columns) > 1:
        # Calcular o desvio padrão para cada categoria (linha)
        df_resultados['variancia'] = df_resultados.std(axis=1)
        
        # Ordenar o DataFrame pela variância para encontrar as maiores divergências
        maiores_diferencas = df_resultados.sort_values(by='variancia', ascending=False).head(7)
        
        console.print("\n[bold magenta]Características Mais Marcantes (Maiores Divergências entre YouTubers)[/bold magenta]")
        
        # Criar tabela
        tabela_diferencas = Table(title="Categorias com Maior Variância de Uso")
        tabela_diferencas.add_column("Categoria LIWC", style="cyan", no_wrap=True)

        # Adicionar as colunas de cada youtuber
        for youtuber in df_resultados.drop(columns=['variancia']).columns:
            tabela_diferencas.add_column(youtuber, style="magenta", justify="right")
        
        # Adicionar coluna do desvio padrão
        tabela_diferencas.add_column("Desvio Padrão", style="yellow", justify="right")

        # Adicionar as linhas da tabela
        for categoria, row_data in maiores_diferencas.iterrows():
            valores_linha = [categoria]

            for youtuber in df_resultados.drop(columns=['variancia']).columns:
                 valores_linha.append(f"{row_data[youtuber]:.2f}%")
            
            valores_linha.append(f"{row_data['variancia']:.2f}")
            tabela_diferencas.add_row(*valores_linha)

        console.print(tabela_diferencas)


if __name__ == '__main__':
    # Definiçoes iniciais
    lista_youtubers = ['Julia MineGirl', 'Tex HS']
    pasta_raiz = 'files'
    pasta_analise = 'analise_palavras_chave'
    caminho_dic_liwc = 'liwc_analysis/LIWC2007_Portugues_win.dic'

    # Carregar as palavras chaves de cada youtuber
    keywords_data = carregar_keywords(lista_youtubers, pasta_raiz, pasta_analise)
    
    # Fazer a análise comparativa das palavras chave
    df_analise_comparativa = analisar_com_liwc(keywords_data, caminho_dic_liwc)
    
    # Mostrar os resultados obtidos
    if df_analise_comparativa is not None:
        apresentar_resultados(df_analise_comparativa)