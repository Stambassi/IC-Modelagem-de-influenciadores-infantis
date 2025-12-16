import pandas as pd
import os
from rich.console import Console
from pathlib import Path
import webbrowser
import spacy
from bertopic import BERTopic

console = Console()

'''
    Função para gerar o lema de uma palavra

    @param palavra - Palavra a ser lematizada
    @param nlp - Modelo de Processamento de Linguagem Natural que faz a lematização
    @return palavr - Palavra lematizada
'''
def lematizar_palavra(palavra, nlp):
    # Verifica se é string e se não está vazia ou só espaços
    if not isinstance(palavra, str) or not palavra.strip():
        return palavra
        
    doc = nlp(palavra)
    
    # Verifica se o Spacy gerou pelo menos um token
    if len(doc) > 0:
        return doc[0].lemma_
    else:
        # Se o Spacy não gerou tokens (ex: era pontuação isolada), retorna original
        return palavra

'''
    Função para lematizar as palavras de um tópico
    @param topic_model - Tópico a ter suas palavras lematizadas
'''
def lematizar_topicos(topic_model):
    # Carrega spacy apenas se necessário
    try:
        nlp = spacy.load("pt_core_news_lg")
    except:
        try:
            nlp = spacy.load("pt_core_news_md")
        except:
            console.print("[red]Spacy não encontrado. Pulando lematização visual.[/red]")
            return

    new_topics = {}
    # Pega apenas tópicos existentes para evitar erros de chave
    topics_ids = [t for t in topic_model.get_topics().keys() if t != -1]

    for topic_id in topics_ids:
        words = topic_model.get_topic(topic_id)
        if not words: continue

        lemmatized = []
        for word, weight in words:
            # Pula se a palavra for vazia ou só espaços
            if not isinstance(word, str) or not word.strip():
                continue

            lemma = lematizar_palavra(word, nlp)
            lemmatized.append((lemma, weight))

        new_topics[topic_id] = lemmatized

    topic_model.topic_repr = new_topics

'''
    Função para carregar o modelo da pasta bertopic/{nome_grupo}/modelo_final e gerar visualizações
    @param nome_grupo - Nome do grupo a ser analisado (Ex: Geral, Minecraft, Roblox)
'''
def visualizar_bertopic(nome_grupo="Geral"):    
    # Define o caminho base modular
    base_dir = Path("bertopic/modelos") / nome_grupo
    model_path = base_dir / "modelo_final"
    csv_path = base_dir / "topicos_info.csv"

    if not model_path.exists():
        console.print(f"[bold red]Erro: Modelo não encontrado em {model_path}[/bold red]")
        console.print("Verifique se você digitou o nome do grupo corretamente ou se já treinou o modelo.")
        return

    try:
        console.print(f"[cyan]Carregando modelo de: {base_dir}[/cyan]")
        topic_model = BERTopic.load(str(model_path))
        
        if csv_path.exists():
            topicos = pd.read_csv(csv_path)
            print(topicos.head())
    except Exception as e:
        console.print(f"[bold red]Erro ao carregar modelo ou CSV: {e}[/bold red]")
        return

    console.print("[yellow]Lematizando tópicos para visualização...[/yellow]")
    lematizar_topicos(topic_model)

    # Contagem de tópicos reais (excluindo o -1 se existir)
    topic_info = topic_model.get_topic_info()
    real_topics_count = len(topic_info[topic_info['Topic'] != -1])
    
    console.print(f"[blue]Tópicos reais identificados: {real_topics_count}[/blue]")

    # ---------------------------------------------------------
    # 1. Gráfico de Barras
    # ---------------------------------------------------------
    try:
        console.print("Gerando Barchart...")
        fig = topic_model.visualize_barchart(top_n_topics=20)
        output_path = base_dir / "topicos_barras.html"
        fig.write_html(output_path)
        webbrowser.open(output_path.resolve().as_uri())
    except Exception as e:
        console.print(f"[red]Erro ao gerar Barchart: {e}[/red]")

    # ---------------------------------------------------------
    # 2. Visualizações Complexas (Requerem > 2 Tópicos)
    # ---------------------------------------------------------
    if real_topics_count > 2:
        # Mapa de Distância
        try:
            console.print("Gerando Intertopic Distance Map...")
            fig = topic_model.visualize_topics()
            output_path = base_dir / "topicos_distancia.html"
            fig.write_html(output_path)
            webbrowser.open(output_path.resolve().as_uri())
        except Exception:
            console.print("[dim]Pufando Mapa de Distância (UMAP instável para poucos tópicos)[/dim]")

        # Heatmap
        try:
            console.print("Gerando Heatmap...")
            fig = topic_model.visualize_heatmap()
            output_path = base_dir / "topicos_heatmap.html"
            fig.write_html(output_path)
            webbrowser.open(output_path.resolve().as_uri())
        except Exception as e:
            console.print(f"[red]Erro no Heatmap: {e}[/red]")

        # Hierarquia
        try:
            console.print("Gerando Hierarquia...")
            fig = topic_model.visualize_hierarchy()
            output_path = base_dir / "topicos_hierarquia.html"
            fig.write_html(output_path)
            webbrowser.open(output_path.resolve().as_uri())
        except Exception as e:
            console.print(f"[red]Erro na Hierarquia: {e}[/red]")
    else:
        console.print("[yellow]Aviso: Tópicos insuficientes para visualizações de Distância, Heatmap e Hierarquia.[/yellow]")

if __name__ == "__main__":
    visualizar_bertopic("Geral")