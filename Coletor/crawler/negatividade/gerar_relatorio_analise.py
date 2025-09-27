import pandas as pd
from pathlib import Path
from detoxify import Detoxify
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

'''
    Função para carregar os arquivos de sentimento e toxicidade de um vídeo e uni-los
    @param video_folder_path - Caminho válido para a pasta de um vídeo
'''
def carregar_e_unificar_dados(video_folder_path: Path) -> pd.DataFrame:
    # Definir o caminho para os arquivos de análise
    path_sentimento = video_folder_path / 'tiras_video.csv'
    path_toxicidade = video_folder_path / 'tiras_video_toxicidade.csv'
    
    # Testar se os arquivos de análise existem
    if not path_sentimento.exists() or not path_toxicidade.exists():
        console.print(f"[red]Erro: Arquivos de análise não encontrados em '{video_folder_path}'[/red]")
        return pd.DataFrame()

    # Ler os arquivos de análise
    df_sent = pd.read_csv(path_sentimento)
    df_toxi = pd.read_csv(path_toxicidade)
    
    # Unir os DataFrames, pegando apenas as colunas de resultado da toxicidade para evitar duplicatas
    colunas_toxicidade = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    df_unificado = pd.concat([df_sent, df_toxi[colunas_toxicidade]], axis=1)
    
    return df_unificado

'''
    Função para calcular e exibir estatísticas agregadas de um vídeo
    @param df_dados_unificados - DataFrame com os dados unificados de sentimento e toxicidade
    @param youtuber - Nome do youtuber para exibir na tela
    @param video_nome - Nome do vídeo para exibir na tela
'''
def gerar_estatisticas_gerais(df_dados_unificados: pd.DataFrame, youtuber: str, video_nome: str) -> None:
    # Testar se o DataFrame é válido
    if df_dados_unificados.empty: return

    # Calcular a correlação em pares das colunas de sentimento negativo e toxicidade
    corr = df_dados_unificados['negatividade'].corr(df_dados_unificados['toxicity'])

    # Calcular as médias de cada análise
    avg_toxicity = df_dados_unificados['toxicity'].mean()
    avg_negativity = df_dados_unificados['negatividade'].mean()

    # Considera uma tira "tóxica" se o score for maior que 0.7 (limiar ajustável)
    percent_toxic_tiras = (df_dados_unificados['toxicity'] > 0.7).sum() / len(df_dados_unificados) * 100

    # Criar o painel de conteúdo da correlação
    panel_content = (
        f"• [b]Toxicidade Média:[/b] {avg_toxicity:.2%}\n"
        f"• [b]Negatividade Média:[/b] {avg_negativity:.2%}\n"
        f"• [b]Tiras Consideradas Tóxicas (>0.7):[/b] {percent_toxic_tiras:.2f}%\n"
        f"• [b]Correlação (Negatividade x Toxicidade):[/b] {corr:.3f}"
    )

    console.print(Panel(panel_content, title=f"Estatísticas Gerais do Vídeo: '{video_nome}' de {youtuber}", border_style="green"))

    # Mostrar a análise geral da correlação
    if corr > 0.5:
        console.print("[italic]Interpretação: Uma correlação alta (>0.5) sugere que, neste vídeo, quando o sentimento é negativo, ele tende a ser também tóxico.[/italic]")
    else:
        console.print("[italic]Interpretação: Uma correlação baixa (<0.5) sugere que negatividade e toxicidade são independentes. O discurso pode ser negativo sem ser tóxico (ex: triste, crítico) e vice-versa.[/italic]")

'''
    Função para encontrar e exibir as tiras com os maiores scores em categorias-chave
    @param df_dados_unificados - DataFrame com os dados unificados de sentimento e toxicidade
'''
def encontrar_tiras_notaveis(df_dados_unificados: pd.DataFrame) -> None:
    # Testar se o DataFrame de dados é válido
    if df_dados_unificados.empty: return
    
    console.print("\n[bold]Análise de Trechos Específicos[/bold]")
    
    # Identificar a mais tóxica
    tira_mais_toxica = df_dados_unificados.loc[df_dados_unificados['toxicity'].idxmax()]

    console.print(Panel(f"'[i]{tira_mais_toxica['tiras']}[/i]'", title="Tira Mais Tóxica do Vídeo", subtitle=f"Score: {tira_mais_toxica['toxicity']:.2%}", border_style="red"))

    # Identificar a tira mais negativa (mas com baixa toxicidade)
    df_dados_unificados_nao_toxico = df_dados_unificados[df_dados_unificados['toxicity'] < 0.5]

    # Testar se não existe tira com negatividade abaixo de 0.5
    if not df_dados_unificados_nao_toxico.empty:
        # Identificar a tira mais negativa, porém não tóxica
        tira_mais_negativa = df_dados_unificados_nao_toxico.loc[df_dados_unificados_nao_toxico['negatividade'].idxmax()]

        console.print(Panel(f"'[i]{tira_mais_negativa['tiras']}[/i]'", title="Tira Mais Negativa (Não-Tóxica)", subtitle=f"Score: {tira_mais_negativa['negatividade']:.2%}", border_style="blue"))
        
    # Identificar a tira mais positiva
    tira_mais_positiva = df_dados_unificados.loc[df_dados_unificados['positividade'].idxmax()]

    console.print(Panel(f"'[i]{tira_mais_positiva['tiras']}[/i]'", title="Tira Mais Positiva do Vídeo", subtitle=f"Score: {tira_mais_positiva['positividade']:.2%}", border_style="magenta"))


'''
    Função para carregar as palavras-chave do TF-IDF e analisar a sua toxicidade
    @param youtuber - Nome do youtuber para mostrar na tela
    @param detoxify_model - Modelo de análise de toxicidade
'''
def analisar_keywords_youtuber(youtuber: str, detoxify_model) -> None:
    console.print("\n--- [bold]Análise de Toxicidade das Palavras-Chave (Perfil do YouTuber)[/bold] ---")
    
    # Criar o caminho para o arquivo de TF-IDF
    path_keywords = Path('files') / youtuber / 'frequencia_palavras' / 'ranking_tfidf.csv'

    # Testar se o arquivo de TF-IDF existe
    if not path_keywords.exists():
        console.print(f"[red]Arquivo de keywords não encontrado para {youtuber}[/red]")
        return
        
    # Ler as palavras-chave do TF-IDF
    df_keywords = pd.read_csv(path_keywords)
    keywords = df_keywords['palavra_chave'].tolist()
    
    # Calcular a toxicidade das palavras
    resultados = detoxify_model.predict(keywords)

    # Transformar os resultados para um DataFrame
    df_resultados_keywords = pd.DataFrame(resultados, index=keywords)
    
    # Identificar as 5 palavras mais tóxicas do ranking
    top_5_toxicas = df_resultados_keywords.sort_values(by='toxicity', ascending=False).head(5)
    
    # Criar a tabela de análise do youtuber
    table = Table(title=f"Nível de Toxicidade das Top Palavras-Chave de {youtuber}")
    table.add_column("Palavra-Chave", style="cyan")
    table.add_column("Score Toxicidade", style="yellow", justify="right")
    
    for palavra, row in top_5_toxicas.iterrows():
        table.add_row(palavra, f"{row['toxicity']:.2%}")
        
    console.print(table)
    avg_toxicity_keywords = df_resultados_keywords['toxicity'].mean()
    console.print(f"Toxicidade média do vocabulário-chave de {youtuber}: [bold]{avg_toxicity_keywords:.2%}[/bold]")


if __name__ == '__main__':
    # Nome do youtuber a ser analisado
    youtuber_name = 'Tex HS'

    # Caminho para a pasta de um vídeo
    video_folder_path = Path(f'files/{youtuber_name}/2020/Abril/NAO ENTRE NA PORTA 806 (Roblox The Lost Episodes)')
    
    # Carregar o modelo Detoxify uma única vez
    console.print("[bold]Carregando modelo Detoxify...[/bold]")
    try:
        model = Detoxify('multilingual')
    except Exception as e:
        console.print(f"Falha ao carregar modelo Detoxify: {e}")
        exit() # Encerra o script se o modelo não carregar

    # Carregar e unir os dados do vídeo
    df_video = carregar_e_unificar_dados(video_folder_path)
    
    if not df_video.empty:
        # Gerar estatísticas gerais
        gerar_estatisticas_gerais(df_video, youtuber_name, video_folder_path.name)
        
        # Encontrar trechos notáveis
        encontrar_tiras_notaveis(df_video)
        
        # Analisar as palavras-chave do YouTuber (perfil geral)
        analisar_keywords_youtuber(youtuber_name, model)