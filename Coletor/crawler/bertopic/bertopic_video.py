import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import webbrowser

from bertopic import BERTopic
from umap import UMAP
import optuna
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


import nltk
from nltk.corpus import stopwords

import spacy

console = Console()

custom_sWords = {"aqui","pra","velho","né","tá","mano","ah",
                 "dela","ju","beleza","jú","julia","olá","tô",
                 "gente","ta","olha","pá","vi","ai","júlia","será",
                 "pessoal","galerinha","acho", "vou", "deus"
                 "daí", "porta","hein","bora","aham","mãe","juma"}

os.environ["TOKENIZERS_PARALLELISM"] = "false"



# -------------------------------------------------------------------
# MÉTRICAS
# -------------------------------------------------------------------

def compute_coherence_and_diversity(topic_model, documents):
    """
    Calcula:
    - Coerência C_v
    - Diversidade de tópicos
    """
    try:
        topics = topic_model.get_topics()

        # Se não houver tópicos suficientes, retorna score ruim
        if not topics:
            return -9999.0, -9999.0

        topic_words = []

        for topic_id in topic_model.get_topics().keys():

            # Ignorar o tópico -1 (outlier)
            if topic_id == -1:
                continue

            words = topic_model.get_topic(topic_id)

            # Ignorar tópicos vazios, None ou formato inesperado
            if not words or not isinstance(words, list):
                continue

            # Filtra apenas pares (token, score)
            valid_pairs = [(w, s) for w, s in words if isinstance(w, str)]

            # Se não houver palavras válidas, ignora
            if len(valid_pairs) == 0:
                continue

            # Pega as top N palavras
            tokens = [w for w, _ in valid_pairs[:10]]

            # Ignora tópicos que ficaram com 0 tokens
            if len(tokens) == 0:
                continue

            topic_words.append(tokens)

        # Se depois de tudo não houver tópicos válidos, retorna score ruim
        if len(topic_words) == 0:
            return -9999.0, -9999.0


        # Coerência C_v
        dictionary = Dictionary(doc.split() for doc in documents)
        texts = [doc.split() for doc in documents]

        print(topic_words[0])

        cm = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()


        # Diversidade de tópicos
        all_words = [w for sub in topic_words for w in sub]
        diversity = len(set(all_words)) / len(all_words)

    except Exception as e:
        # console.log(e,log_locals=True)
        print("Erro:", e)
        return -9999.0,-9999.0

    return coherence, diversity


# -------------------------------------------------------------------
# FUNÇÃO OBJETIVA PARA OTIMIZAR COM OPTUNA
# -------------------------------------------------------------------

def objective(trial, documents):

    # ---------------------------
    # Hiperparâmetros do UMAP
    # ---------------------------
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
    n_components = trial.suggest_int("n_components", 5, 15)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
    )

    # ---------------------------
    # Hiperparâmetros do HDBSCAN
    # ---------------------------
    min_cluster_size = trial.suggest_int("min_cluster_size", 2, 20)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True
    )

    # ---------------------------
    # Vetorizador
    # ---------------------------
    n_docs = len(documents)

    min_df = trial.suggest_float("min_df", 0.0, 0.3)
    max_df = trial.suggest_float("max_df", 1, 1)

    if max_df < (min_df / n_docs):
        trial.set_user_attr("skip_reason", "max_df < min_df at corpus level")
        return -9999.0

    # Stopwords
    nlp = spacy.load("pt_core_news_md")
    spacy_sWords = nlp.Defaults.stop_words
    stop_words = list(spacy_sWords.union(custom_sWords))

    vectorizer_model = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words
    )

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # ---------------------------
    # Criar modelo
    # ---------------------------
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=False
    )

    # ---------------------------
    # Rodar BERTopic
    # ---------------------------
    topics, probs = topic_model.fit_transform(documents)

    # ---------------------------
    # Avaliar qualidade
    # ---------------------------
    coherence, diversity = compute_coherence_and_diversity(topic_model, documents)

    final_score = 0.7 * coherence + 0.3 * diversity

        # Log no trial
    trial.set_user_attr("coherence", coherence)
    trial.set_user_attr("diversity", diversity)


    
    return final_score


# -------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------------------------------------------

def otimizar_BERTopic(documents, n_trials=30):

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(lambda trial: objective(trial, documents), n_trials=n_trials)

    print("\n🎯 Melhor score:", study.best_value)
    print("📌 Melhores hiperparâmetros:")
    print(study.best_params)

    return study


def lematizar_json_segment(video_data: json) -> list[str]:
    texto_limpo = []
    for segment in video_data['segments']:
        documento = nlp(segment['text'])
        tokens_segmento = []
        for token in documento:
            if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
                if preservar_tamanho:
                    tokens_segmento.append(str.lower(token.lemma_))
                else:
                    texto_limpo.append(str.lower(token.lemma_))
            elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
                if preservar_tamanho:
                    tokens_segmento.append(str.lower(token.text))
                else:
                    texto_limpo.append(str.lower(token.text))
        if preservar_tamanho: 
            texto_limpo.append(" ".join(tokens_segmento))
    return texto_limpo

def lematizar_list(data: list) -> list[str]:
    # print(data)

    ## filtrar palavras por frequenicia
    X = vectorizer_model.fit_transform(data)
    filtered_words = vectorizer_model.get_feature_names_out()

    documento = nlp(" ".join(filtered_words))

    # result = ""
    # for item in data:
    #     result += item
    # documento = nlp(result)

    texto_limpo = []
    for token in documento:
        if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.lemma_))
        elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.text))
    
    return texto_limpo

def lematizar_json(video_data: json) -> list[str]:
    documento = nlp(video_data['text'])
    texto_limpo = []
    for token in documento:
        if token.pos_ == "VERB" and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.lemma_))
        elif not (token.pos_ == "VERB") and token.is_alpha and not (token.is_stop):
            texto_limpo.append(str.lower(token.text))
    return texto_limpo

def coletar_informacoes_youtuber(video_path) -> str: 
    try:
        with open(video_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = video_data['text']
    except:
        texto_limpo = ""
    return texto_limpo

def coletar_tirinhas_video(tirinha_csv_path) -> list[str]:
    try:
        sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
        sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()

        tiras_csv = pd.read_csv(tirinha_csv_path)
        tiras = tiras_csv['tiras'].tolist()

        indices_selecionados = set()

        for i, estado in enumerate(sequencia):
            if estado == "T":
                inicio = max(0, i - 3)
                fim = min(len(tiras) - 1, i + 3)
                for j in range(inicio, fim + 1):
                    indices_selecionados.add(j)

        tirinhas_coletadas = [tiras[i] for i in sorted(indices_selecionados)]

        return " ".join(tirinhas_coletadas)

    except Exception as e:
        console.log(e)
        return ""


def get_dados_youtuber(youtuber: str, arquivo_tirinha = 'tiras_video.csv'):
    documento = []
    base_path = Path(f'files/{youtuber}')
    for tirinha_csv_path in base_path.rglob(arquivo_tirinha):
        tiras_youtuber = coletar_tirinhas_video(tirinha_csv_path)
        if tiras_youtuber != "":
            documento.append(tiras_youtuber)
    return documento
    
def pipeline_BERTopic(youtuber: str):
    documentos = get_dados_youtuber(youtuber)
    
    console.rule(f"Documento de tamanho: {len(documentos)}")
    print(type(documentos))
    study = otimizar_BERTopic(documentos, n_trials=20)
    salvar_BERTopic(documentos, study.best_params, youtuber)



def salvar_BERTopic(docs, params, youtuber):
    
    ## Embedding
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    ## Reduzir dimensionalidade
    umap_model = UMAP(n_neighbors=params['n_neighbors'], 
                      n_components=params['n_components'], 
                      min_dist=params['min_dist'],
                      metric='euclidean',
                      random_state=42)

    ## Ajuste no Cluster
    hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'], 
                            metric='euclidean',
                            cluster_selection_method='leaf',
                            prediction_data=True)

    ## Tokenizar
    nlp = spacy.load("pt_core_news_md")

    # nltk.download("stopwords")
    # stop_words = stopwords.words("portuguese")
    spacy_sWords = nlp.Defaults.stop_words

    stop_words = list(spacy_sWords.union(custom_sWords))

    vectorizer_model = CountVectorizer(min_df=params['min_df'], 
                                    #    max_df=study['max_df'],
                                       stop_words=stop_words)
    # vectorizer_model = CountVectorizer()

    ## Representação dos topicos
    ctfidf_model = ClassTfidfTransformer()

    # topic_model = BERTopic(language="portuguese",vectorizer_model = vectorizer_model)
    # topic_model = BERTopic(language="portuguese")

    topic_model = BERTopic(
    embedding_model=embedding_model,          
    umap_model=umap_model,                    
    hdbscan_model=hdbscan_model,              
    vectorizer_model=vectorizer_model,        
    ctfidf_model=ctfidf_model,                
    )
    # Comando para baixar o modelo
    # $ python -m spacy download pt_core_news_md
    
    console.rule(f"Gerando Tópicos com {len(docs)} tirinhas")
    topics, probs = topic_model.fit_transform(docs)
    topicos = topic_model.get_topic_info()
    topicos.to_csv(f"bertopic/topicos_{youtuber}.csv", index=False)
    topic_model.save(f"bertopic/modelo_{youtuber}")



def gerar_topicos_youtubers(youtubers_list: list[str]) -> None:

    ###### Fine Tuning BERTopic
    console.print("Ajustando hiperparâmetros do BERTopic...")

    ## Embedding
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    ## Reduzir dimensionalidade
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='euclidean', random_state=42)

    ## Ajuste no Cluster
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)

    ## Tokenizar
    nlp = spacy.load("pt_core_news_md")

    # nltk.download("stopwords")
    # stop_words = stopwords.words("portuguese")
    spacy_sWords = nlp.Defaults.stop_words

    stop_words = list(spacy_sWords.union(custom_sWords))

    vectorizer_model = CountVectorizer(min_df=2, stop_words=stop_words)
    # vectorizer_model = CountVectorizer()

    ## Representação dos topicos
    ctfidf_model = ClassTfidfTransformer()

    # topic_model = BERTopic(language="portuguese",vectorizer_model = vectorizer_model)
    # topic_model = BERTopic(language="portuguese")

    topic_model = BERTopic(
    embedding_model=embedding_model,          
    umap_model=umap_model,                    
    hdbscan_model=hdbscan_model,              
    vectorizer_model=vectorizer_model,        
    ctfidf_model=ctfidf_model,                
    )
    # Comando para baixar o modelo
    # $ python -m spacy download pt_core_news_md
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        documento = get_dados_youtuber(youtuber)
        # print(tiras_youtuber)
        console.rule(f"Gerando Tópicos com {len(documento)} tirinhas")
        topics, probs = topic_model.fit_transform(documento)
        topicos = topic_model.get_topic_info()
        topicos.to_csv(f"bertopic/topicos_{youtuber}.csv", index=False)
        topic_model.save(f"bertopic/modelo_{youtuber}")

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


def visualizar_bertopic(youtuber):
    nlp = spacy.load("pt_core_news_lg")

    topic_model = BERTopic.load(f"bertopic/modelo_{youtuber}")
    topicos = pd.read_csv(f"bertopic/topicos_{youtuber}.csv")

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



# lista_youtubers =  ['Amy Scarlet', 'Julia MineGirl', 'Lokis', 'Luluca Games', 'meu nome é david', 'Papile','Tex HS']

# lista_youtubers =  ['AuthenticGames']
lista_youtubers =  ['Julia MineGirl']

pipeline_BERTopic('Julia MineGirl')
# visualizar_bertopic("Julia MineGirl")



    