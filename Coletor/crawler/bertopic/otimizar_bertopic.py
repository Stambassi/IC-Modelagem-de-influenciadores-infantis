import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import numpy as np # Necessário para cálculos
import spacy

from bertopic import BERTopic
from umap import UMAP
import optuna
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import PartOfSpeech, KeyBERTInspired

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

console = Console()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Carregamento global do Spacy para evitar reload no loop
try:
    nlp_engine = spacy.load("pt_core_news_lg")
except:
    console.print("[yellow]Aviso: pt_core_news_lg não encontrado, tentando carregar pt_core_news_md ou sm[/yellow]")
    try:
        nlp_engine = spacy.load("pt_core_news_md")
    except:
        nlp_engine = spacy.load("pt_core_news_sm")


# -------------------------------------------------------------------
# MÉTRICAS
# -------------------------------------------------------------------

def compute_metrics(topic_model, documents):
    """
    Calcula:
    1. Coerência (C_v)
    2. Diversidade de Tópicos (Proporção de palavras únicas nos top-n)
    3. Taxa de Ruído (Outliers)
    """
    try:
        topic_info = topic_model.get_topic_info()
        
        # Verificar Ruído (Tópico -1)
        outlier_row = topic_info[topic_info['Topic'] == -1]
        outlier_count = outlier_row['Count'].values[0] if not outlier_row.empty else 0
        noise_ratio = outlier_count / len(documents)

        # Se só existe o tópico -1 ou nenhum tópico, falha.
        if len(topic_info) <= 1:
            return -9999.0, -9999.0, 1.0

        # Extração de palavras para Coerência e Diversidade
        topic_words = []
        all_top_words = []

        topics_dict = topic_model.get_topics()
        
        for topic_id in topics_dict:
            if topic_id == -1: continue # Pula ruído

            words_probs = topics_dict[topic_id]
            if not words_probs: continue
            
            # Pega as top 10 palavras do tópico
            tokens = [w for w, _ in words_probs[:10] if isinstance(w, str)]
            
            if tokens:
                topic_words.append(tokens)
                all_top_words.extend(tokens)

        if not topic_words:
            return -9999.0, -9999.0, 1.0

        # 1. Coerência C_v
        # Nota: Idealmente, tokenizar os docs apenas uma vez fora da função, mas mantendo aqui por simplicidade
        texts = [doc.split() for doc in documents]
        dictionary = Dictionary(texts)
        
        cm = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()

        # 2. Diversidade de Tópicos (Topic Diversity)
        # Calcula quantos % das palavras nos top-10 de todos os tópicos são únicas
        # Se 1.0 (100%), todos os tópicos têm palavras diferentes. Se 0, são todos iguais.
        if len(all_top_words) > 0:
            diversity = len(set(all_top_words)) / len(all_top_words)
        else:
            diversity = 0

        return coherence, diversity, noise_ratio

    except Exception as e:
        console.print(f"[red]Erro no cálculo de métricas: {e}[/red]")
        return -9999.0, -9999.0, 1.0


# -------------------------------------------------------------------
# FUNÇÃO OBJETIVA PARA OTIMIZAR COM OPTUNA
# -------------------------------------------------------------------
def make_objective(documents, param_ranges, stop_words):

    def objective(trial):

        # ---------------------------
        # A. Hiperparâmetros
        # ---------------------------
        
        # UMAP
        n_neighbors = trial.suggest_int("n_neighbors", param_ranges["n_neighbors"]["low"], param_ranges["n_neighbors"]["high"])
        n_components = trial.suggest_int("n_components", param_ranges["n_components"]["low"], param_ranges["n_components"]["high"])
        min_dist = trial.suggest_float("min_dist", param_ranges["min_dist"]["low"], param_ranges["min_dist"]["high"])

        # HDBSCAN
        min_cluster_size = trial.suggest_int("min_cluster_size", param_ranges["min_cluster_size"]["low"], param_ranges["min_cluster_size"]["high"])
        
        # Restrição Lógica: min_samples deve ser <= min_cluster_size
        # Definimos o teto do min_samples dinamicamente
        high_samples = min(min_cluster_size, param_ranges["min_samples"]["high"])
        low_samples = min(param_ranges["min_samples"]["low"], high_samples) # Segurança caso low > high temporariamente
        
        min_samples = trial.suggest_int("min_samples", low_samples, high_samples)

        # Vectorizer
        # MUDANÇA: suggest_int para min_df (limpeza absoluta, não percentual)
        min_df_val = param_ranges["min_df"]
        if min_df_val["type"] == "int":
             min_df = trial.suggest_int("min_df", min_df_val["low"], min_df_val["high"])
        else:
             # Fallback caso venha como float do pipeline antigo, mas recomendamos int
             min_df = trial.suggest_float("min_df", min_df_val["low"], min_df_val["high"])

        ngram_range = trial.suggest_categorical("ngram_range", param_ranges["ngram_range"]["choices"])


        # ---------------------------
        # B. Construção do Modelo
        # ---------------------------
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
            cluster_selection_method="eom", # 'eom' costuma ser melhor que 'leaf' para NLP, mas pode manter leaf
            prediction_data=True
        )

        vectorizer_model = CountVectorizer(
            min_df=min_df,
            max_df=1.0, # Fixo, pois já tratamos stop words
            ngram_range=ngram_range,
            stop_words=stop_words
        )

        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        representation_model = KeyBERTInspired()

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            verbose=False
        )

        # ---------------------------
        # C. Treinamento e Avaliação
        # ---------------------------
        try:
            topic_model.fit_transform(documents)
        except Exception as e:
            # Caso o UMAP/HDBSCAN falhe (ex: dataset muito pequeno para os parâmetros)
            return -9999.0

        coherence, diversity, noise_ratio = compute_metrics(topic_model, documents)

        # ---------------------------
        # D. Penalização (Guardrails)
        # ---------------------------
        # Cálculo do Score Base
        base_score = (0.7 * coherence) + (0.3 * diversity)
        
        # Penalização Suave por Ruído
        # Se 50% for ruído, o score cai pela metade.
        if noise_ratio > 0.0:
            final_score = base_score * (1.0 - noise_ratio)
        else:
            final_score = base_score

        # Logs para o Optuna
        trial.set_user_attr("coherence", coherence)
        trial.set_user_attr("diversity", diversity)
        trial.set_user_attr("noise_ratio", noise_ratio)
        trial.set_user_attr("n_topics", len(topic_model.get_topic_info()) - 1)
        
        # Se o score for muito baixo devido à penalidade, ainda retornamos ele
        # para o Optuna aprender o caminho, em vez de retornar erro.
        return max(final_score, 0.0)

    return objective

# -------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -------------------------------------------------------------------

def otimizar_BERTopic(documents, param_ranges, stop_words, n_trials=30):

    # Garante que stop_words seja uma lista
    if isinstance(stop_words, set):
        stop_words = list(stop_words)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    objective = make_objective(documents, param_ranges, stop_words)

    console.print(f"[yellow]Iniciando otimização com {n_trials} trials...[/yellow]")
    study.optimize(objective, n_trials=n_trials)

    print("\n🎯 Melhor score:", study.best_value)
    print("📌 Melhores hiperparâmetros:", study.best_params)

    return study


def salvar_BERTopic(docs, params, stop_words, nome_arquivo_extra=""):
    
    console.rule("Salvando Melhor Modelo")

    if isinstance(stop_words, set):
        stop_words = list(stop_words)

    # Reconstrução exata do pipeline com os melhores parâmetros
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    umap_model = UMAP(
        n_neighbors=params['n_neighbors'], 
        n_components=params['n_components'], 
        min_dist=params['min_dist'],
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=params['min_cluster_size'], 
        min_samples=params['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom', # Consistente com a otimização
        prediction_data=True
    )

    # Conversão segura de tipos para o Vectorizer
    min_df = int(params['min_df']) if isinstance(params['min_df'], float) and params['min_df'] > 1 else params['min_df']
    
    vectorizer_model = CountVectorizer(
        min_df=min_df, 
        max_df=1.0,
        ngram_range=params['ngram_range'],
        stop_words=stop_words
    )

    ctfidf_model = ClassTfidfTransformer()
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,          
        umap_model=umap_model,                    
        hdbscan_model=hdbscan_model,              
        vectorizer_model=vectorizer_model,        
        ctfidf_model=ctfidf_model,
        representation_model=representation_model
    )
    
    topic_model.fit_transform(docs)
    
    # Salvar outputs
    os.makedirs("bertopic", exist_ok=True)
    
    topicos = topic_model.get_topic_info()
    topicos.to_csv(f"bertopic/topicos_{nome_arquivo_extra}.csv", index=False)
    
    topic_model.save(f"bertopic/modelo_{nome_arquivo_extra}", serialization="safetensors")
    
    console.print("[bold green]Modelo salvo com sucesso na pasta /bertopic![/bold green]")