import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import webbrowser

from bertopic import BERTopic


import nltk
from nltk.corpus import stopwords

import spacy

console = Console()

custom_sWords = {"aqui","pra","velho","né","tá","mano","ah",
                 "dela","ju","beleza","jú","julia","olá","tô",
                 "gente","ta","olha","pá","vi","ai","júlia","será",
                 "pessoal","galerinha","acho", "vou", 
                 "daí", "porta","hein","bora","aham","juma"}



def lematizar_palavra(palavra, nlp):
    doc = nlp(palavra)
    return doc[0].lemma_

def lematizar_topicos(topic_model):
    nlp = spacy.load("pt_core_news_lg")
    new_topics = {}

    for topic_id in topic_model.get_topics().keys():
        if topic_id == -1:   # tópico outlier do BERTopic
            continue
        
        words = topic_model.get_topic(topic_id)
        lemmatized = []

        for word, weight in words:
            lemma = lematizar_palavra(word,nlp)
            lemmatized.append((lemma, weight))

        new_topics[topic_id] = lemmatized

    # alteramos os tópicos apenas para visualização
    topic_model.topic_repr = new_topics


def visualizar_bertopic(nome_arquivo_extra=""):
    nlp = spacy.load("pt_core_news_lg")

    topic_model = BERTopic.load(f"bertopic/modelo_{nome_arquivo_extra}")
    topicos = pd.read_csv(f"bertopic/topicos_{nome_arquivo_extra}.csv")

    lematizar_topicos(topic_model)
    print(topicos)

    fig = topic_model.visualize_heatmap()
    heatmap_path = Path("bertopic/topicos_heatmap.html")
    fig.write_html(heatmap_path)

    fig = topic_model.visualize_topics()
    topics_path = Path("bertopic/topicos.html")
    fig.write_html(topics_path)

    fig = topic_model.visualize_hierarchy()
    hierarchy_path = Path("bertopic/topicos_hierarquia.html")
    fig.write_html(hierarchy_path)

    # abrir automaticamente
    webbrowser.open(heatmap_path.resolve().as_uri())
    webbrowser.open(topics_path.resolve().as_uri())
    webbrowser.open(hierarchy_path.resolve().as_uri())

