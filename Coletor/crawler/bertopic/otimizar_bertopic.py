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
from bertopic.representation import PartOfSpeech, KeyBERTInspired

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


import nltk
from nltk.corpus import stopwords

import spacy

console = Console()

custom_sWords = {"aqui","pra","velho","né","tá","mano","ah",
                 "dela","ju","beleza","jú","julia","olá","tô",
                 "gente","ta","olha","pá","vi","ai","júlia","será",
                 "pessoal","galerinha","acho", "vou", 
                 "daí", "porta","hein","bora","aham","juma"}

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
            console.print("[red]ERRO:[/] Topics vazio")
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
            console.print("[red]ERRO:[/] Topic_words vazio")
            print(topics)

            return -9999.0, -9999.0


        # Coerência C_v
        dictionary = Dictionary(doc.split() for doc in documents)
        texts = [doc.split() for doc in documents]


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
def make_objective(documents, param_ranges):

    def objective(trial):

        # ---------------------------
        # Hiperparâmetros do UMAP
        # ---------------------------
        n_neighbors = trial.suggest_int(
            "n_neighbors",
            param_ranges["n_neighbors"]["low"],
            param_ranges["n_neighbors"]["high"]
        )

        n_components = trial.suggest_int(
            "n_components",
            param_ranges["n_components"]["low"],
            param_ranges["n_components"]["high"]
        )

        min_dist = trial.suggest_float(
            "min_dist",
            param_ranges["min_dist"]["low"],
            param_ranges["min_dist"]["high"]
        )

        # ---------------------------
        # Hiperparâmetros do HDBSCAN
        # ---------------------------
        min_cluster_size = trial.suggest_int(
            "min_cluster_size",
            param_ranges["min_cluster_size"]["low"],
            param_ranges["min_cluster_size"]["high"]
        )

        min_samples = trial.suggest_int(
            "min_samples",
            param_ranges["min_samples"]["low"],
            param_ranges["min_samples"]["high"]
        )

        # ---------------------------
        # Vetorizador
        # ---------------------------

        min_df = trial.suggest_float(
            "min_df",
            param_ranges["min_df"]["low"],
            param_ranges["min_df"]["high"]
        )

        max_df = trial.suggest_float(
            "max_df",
            param_ranges["max_df"]["low"],
            param_ranges["max_df"]["high"]
        )

        ngram_range = trial.suggest_categorical(
            "ngram_range",
            param_ranges["ngram_range"]["choices"]
        )

        umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True
        )

        n_docs = len(documents)

        if max_df < (min_df / n_docs):
            console.print("[red]ERRO:[/] max_df < min_df at corpus level")
            trial.set_user_attr("skip_reason", "max_df < min_df at corpus level")
            return -9999.0

        # Stopwords
        nlp = spacy.load("pt_core_news_lg")
        spacy_sWords = nlp.Defaults.stop_words
        stop_words = list(spacy_sWords.union(custom_sWords))

        vectorizer_model = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words
        )

        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


        # Modelo de Finetuning
        representation_model = PartOfSpeech("pt_core_news_lg")
        # representation_model = KeyBERTInspired()

        # ---------------------------
        # Criar modelo
        # ---------------------------
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
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

    return objective





    


# -------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------------------------------------------

def otimizar_BERTopic(documents, param_ranges, n_trials=30):

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    objective = make_objective(documents, param_ranges)

    study.optimize(objective, n_trials=n_trials)

    print("\n🎯 Melhor score:", study.best_value)
    print("📌 Melhores hiperparâmetros:")
    print(study.best_params)

    return study



def salvar_BERTopic(docs, params, nome_arquivo_extra = ""):
    
    ## Embedding
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    ## Reduzir dimensionalidade
    umap_model = UMAP(n_neighbors=params['n_neighbors'], 
                      n_components=params['n_components'], 
                      min_dist=params['min_dist'],
                      metric='cosine',
                      random_state=42)

    ## Ajuste no Cluster
    hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'], 
                            min_samples=params['min_samples'],
                            metric='euclidean',
                            cluster_selection_method='leaf',
                            prediction_data=True)

    ## Tokenizar
    nlp = spacy.load("pt_core_news_lg")

    # nltk.download("stopwords")
    # stop_words = stopwords.words("portuguese")
    spacy_sWords = nlp.Defaults.stop_words

    stop_words = list(spacy_sWords.union(custom_sWords))

    vectorizer_model = CountVectorizer(min_df=params['min_df'], 
                                       max_df=params['max_df'],
                                       ngram_range=params['ngram_range'],
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
    topicos.to_csv(f"bertopic/topicos_{nome_arquivo_extra}.csv", index=False)
    topic_model.save(f"bertopic/modelo_{nome_arquivo_extra}")



    