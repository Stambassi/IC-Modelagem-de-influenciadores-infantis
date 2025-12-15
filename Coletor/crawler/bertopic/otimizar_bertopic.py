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

        print(f"Quantidade de tópicos encontrados: {len(topic_info) - 1}") # -1 para descontar o outlier
        print(topic_info.head())
        
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
# -------------------------------------------------------------------
# FUNÇÃO OBJETIVA PARA OTIMIZAR COM OPTUNA (ATUALIZADA)
# -------------------------------------------------------------------
# def make_objective(documents, param_ranges, stop_words):

#     def objective(trial):

#         # ---------------------------
#         # A. Hiperparâmetros
#         # ---------------------------
#         n_neighbors = trial.suggest_int("n_neighbors", param_ranges["n_neighbors"]["low"], param_ranges["n_neighbors"]["high"])
#         n_components = trial.suggest_int("n_components", param_ranges["n_components"]["low"], param_ranges["n_components"]["high"])
#         min_dist = trial.suggest_float("min_dist", param_ranges["min_dist"]["low"], param_ranges["min_dist"]["high"])
        
#         min_cluster_size = trial.suggest_int("min_cluster_size", param_ranges["min_cluster_size"]["low"], param_ranges["min_cluster_size"]["high"])
        
#         # Garante min_samples <= min_cluster_size
#         high_samples = min(min_cluster_size, param_ranges["min_samples"]["high"])
#         low_samples = min(param_ranges["min_samples"]["low"], high_samples)
#         min_samples = trial.suggest_int("min_samples", low_samples, high_samples)

#         # Min_df como inteiro
#         min_df_val = param_ranges["min_df"]
#         if min_df_val["type"] == "int":
#              min_df = trial.suggest_int("min_df", min_df_val["low"], min_df_val["high"])
#         else:
#              min_df = trial.suggest_float("min_df", min_df_val["low"], min_df_val["high"])

#         ngram_range = trial.suggest_categorical("ngram_range", param_ranges["ngram_range"]["choices"])

#         # ---------------------------
#         # B. Construção do Modelo
#         # ---------------------------
#         try:
#             umap_model = UMAP(
#                 n_neighbors=n_neighbors,
#                 n_components=n_components,
#                 min_dist=min_dist,
#                 metric="cosine",
#                 random_state=42
#             )

#             hdbscan_model = HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 metric="euclidean",
#                 cluster_selection_method="eom", 
#                 prediction_data=True
#             )

#             vectorizer_model = CountVectorizer(
#                 min_df=min_df,
#                 max_df=1.0, 
#                 ngram_range=ngram_range,
#                 stop_words=stop_words
#             )

#             embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#             representation_model = KeyBERTInspired()

#             topic_model = BERTopic(
#                 embedding_model=embedding_model,
#                 umap_model=umap_model,
#                 hdbscan_model=hdbscan_model,
#                 vectorizer_model=vectorizer_model,
#                 representation_model=representation_model,
#                 verbose=False
#             )

#             # ---------------------------
#             # C. Treinamento
#             # ---------------------------
#             topic_model.fit_transform(documents)
            
#         except Exception as e:
#             # Captura erros de UMAP/HDBSCAN (ex: dataset muito pequeno para n_neighbors)
#             return -9999.0

#         # ---------------------------
#         # D. Cálculo de Métricas Base
#         # ---------------------------
#         coherence, diversity, noise_ratio = compute_metrics(topic_model, documents)
        
#         # Se falhou nas métricas, retorna erro
#         if coherence == -9999.0:
#             return -9999.0

#         # Score de Qualidade Pura
#         quality_score = (0.7 * coherence) + (0.3 * diversity)

#         # ---------------------------
#         # E. Bônus por Quantidade de Tópicos (NOVA LÓGICA)
#         # ---------------------------
#         topic_info = topic_model.get_topic_info()
#         n_topics = len(topic_info) - 1  # Subtrai o tópico -1 (ruído)
        
#         # Fator de recompensa: 0.02 pontos por tópico extra
#         # Limitado (cap) a 0.20 (equivalente a 10 tópicos). 
#         # Isso evita que o modelo crie 50 tópicos ruins só para ganhar pontos.
#         topic_bonus = min(n_topics * 0.02, 0.20)
        
#         # Se tiver MENOS de 3 tópicos, aplicamos uma penalidade extra
#         # para desencorajar fortemente modelos de 2 tópicos
#         if n_topics < 3:
#             topic_bonus -= 0.10

#         # ---------------------------
#         # F. Penalização de Ruído e Score Final
#         # ---------------------------
        
#         # Soma o bônus à qualidade base
#         score_with_bonus = quality_score + topic_bonus
        
#         # Aplica a penalidade de ruído (multiplicativo)
#         # Se ruído for 0.5 (50%), o score cai pela metade.
#         if noise_ratio > 0.45:
#              # Penaliza severamente se passar do limite aceitável
#              final_score = -1.0 * noise_ratio
#              trial.set_user_attr("skipped", "high_noise")
#         else:
#              # Penaliza proporcionalmente
#              final_score = score_with_bonus * (1.0 - noise_ratio)

#         # Logs para análise no Optuna
#         trial.set_user_attr("n_topics", n_topics)
#         trial.set_user_attr("coherence", coherence)
#         trial.set_user_attr("diversity", diversity)
#         trial.set_user_attr("noise_ratio", noise_ratio)
#         trial.set_user_attr("raw_quality", quality_score)

#         return max(final_score, -1.0) # Garante retorno válido

#     return objective

def make_objective(documents, param_ranges, stop_words):

    def objective(trial):

        # ---------------------------
        # A. Hiperparâmetros
        # ---------------------------
        n_neighbors = trial.suggest_int("n_neighbors", param_ranges["n_neighbors"]["low"], param_ranges["n_neighbors"]["high"])
        n_components = trial.suggest_int("n_components", param_ranges["n_components"]["low"], param_ranges["n_components"]["high"])
        min_dist = trial.suggest_float("min_dist", param_ranges["min_dist"]["low"], param_ranges["min_dist"]["high"])
        
        min_cluster_size = trial.suggest_int("min_cluster_size", param_ranges["min_cluster_size"]["low"], param_ranges["min_cluster_size"]["high"])
        
        # Garante min_samples <= min_cluster_size
        high_samples = min(min_cluster_size, param_ranges["min_samples"]["high"])
        low_samples = min(param_ranges["min_samples"]["low"], high_samples)
        min_samples = trial.suggest_int("min_samples", low_samples, high_samples)

        # Min_df como inteiro
        min_df_val = param_ranges["min_df"]
        if min_df_val["type"] == "int":
             min_df = trial.suggest_int("min_df", min_df_val["low"], min_df_val["high"])
        else:
             min_df = trial.suggest_float("min_df", min_df_val["low"], min_df_val["high"])

        ngram_range = trial.suggest_categorical("ngram_range", param_ranges["ngram_range"]["choices"])

        # ---------------------------
        # B. Construção do Modelo
        # ---------------------------
        try:
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
                cluster_selection_method="eom", 
                prediction_data=True
            )

            vectorizer_model = CountVectorizer(
                min_df=min_df,
                max_df=1.0, 
                ngram_range=ngram_range,
                stop_words=stop_words,
                token_pattern=r'(?u)\b\w\w+\b',
                strip_accents='unicode'
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
            # C. Treinamento
            # ---------------------------
            topic_model.fit_transform(documents)
            
        except Exception as e:
            # Captura erros de UMAP/HDBSCAN (ex: dataset muito pequeno para n_neighbors)
            return -9999.0

        # ---------------------------
        # D. Cálculo de Métricas Base
        # ---------------------------
        coherence, diversity, noise_ratio = compute_metrics(topic_model, documents)
        
        # Se falhou nas métricas, retorna erro
        if coherence == -9999.0:
            return -9999.0

        # MUDANÇA AQUI: Usamos APENAS a Coerência como score base de qualidade
        # Removemos a média ponderada com diversidade.
        quality_score = coherence

        # ---------------------------
        # E. Bônus por Quantidade de Tópicos
        # ---------------------------
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # Subtrai o tópico -1 (ruído)
        
        # Fator de recompensa: 0.02 pontos por tópico extra (Cap em 0.20)
        topic_bonus = min(n_topics * 0.02, 0.20)
        
        # Penalidade extra se tiver menos de 3 tópicos (para evitar underfitting)
        if n_topics < 3:
            topic_bonus -= 0.10

        # ---------------------------
        # F. Penalização de Ruído e Score Final
        # ---------------------------
        
        # Soma o bônus à qualidade base
        score_with_bonus = quality_score + topic_bonus
        
        # Aplica a penalidade de ruído (multiplicativo)
        if noise_ratio > 0.45:
             final_score = -1.0 * noise_ratio
             trial.set_user_attr("skipped", "high_noise")
        else:
             final_score = score_with_bonus * (1.0 - noise_ratio)

        # Logs para análise no Optuna
        trial.set_user_attr("n_topics", n_topics)
        trial.set_user_attr("coherence", coherence)
        trial.set_user_attr("diversity", diversity) # Mantemos no log apenas para curiosidade
        trial.set_user_attr("noise_ratio", noise_ratio)
        trial.set_user_attr("raw_quality", quality_score)

        return max(final_score, -1.0) 

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


'''
    Reconstrói o modelo com os melhores parâmetros, treina e salva 
    os artefatos (Modelo, CSV e Gráficos) em uma pasta específica do grupo.
'''
def salvar_BERTopic(docs, params, stop_words, nome_grupo="Geral"):
    
    console.rule(f"Salvando Melhor Modelo para: {nome_grupo}")

    # Definição da pasta de saída
    # Ex: bertopic/Minecraft/ ou bertopic/Geral/
    output_dir = os.path.join("bertopic/modelos", nome_grupo)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(stop_words, set):
        stop_words = list(stop_words)

    # Reconstrução do Pipeline
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
        cluster_selection_method='eom', 
        prediction_data=True
    )

    # Conversão segura de tipos
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
        representation_model=representation_model,
        nr_topics=None
    )
    
    # Treinamento final
    console.print(f"Treinando modelo base com {len(docs)} documentos...")
    topic_model.fit_transform(docs)
    
    # Redução de tópicos segura
    # Antes de reduzir, altera o min_df para não quebrar o c-TF-IDF dos tópicos fundidos
    console.print("Aplicando redução de tópicos (nr_topics='auto')...")
    
    # Relaxa a restrição para evitar o erro "min_df > n_topics"
    topic_model.vectorizer_model.min_df = 1 
    
    # Reduz os tópicos com segurança
    topic_model.reduce_topics(docs, nr_topics="auto")

    # Salvando Arquivos
    console.print("Salvando artefatos...")
    
    # Salvar CSV de informações
    topicos = topic_model.get_topic_info()
    csv_path = os.path.join(output_dir, "topicos_info.csv")
    topicos.to_csv(csv_path, index=False)
    
    # Salvar o Modelo (formato safetensors é mais seguro e rápido)
    model_path = os.path.join(output_dir, "modelo_final")
    topic_model.save(model_path, serialization="safetensors")
    
    # Gerando e Salvando Gráficos
    console.print("Gerando gráficos de visualização...")

    # Gráfico de Barras (Sempre funciona se tiver tópicos)
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=20)
        fig_bar.write_html(os.path.join(output_dir, "grafico_barras.html"))
    except Exception as e:
        console.print(f"[yellow]Aviso: Não foi possível gerar gráfico de barras ({e})[/yellow]")

    # Hierarquia e Heatmap (Requerem > 2 tópicos reais)
    # Conta tópicos ignorando o -1 (ruído)
    n_topics = len([t for t in topic_model.get_topics().keys() if t != -1])
    
    if n_topics > 2:
        try:
            fig_hier = topic_model.visualize_hierarchy()
            fig_hier.write_html(os.path.join(output_dir, "grafico_hierarquia.html"))
            
            fig_heat = topic_model.visualize_heatmap()
            fig_heat.write_html(os.path.join(output_dir, "grafico_heatmap.html"))
        except Exception as e:
            console.print(f"[yellow]Aviso: Erro ao gerar visualizações complexas ({e})[/yellow]")
    else:
        console.print("[dim]Pufando Hierarquia e Heatmap (menos de 3 tópicos identificados)[/dim]")

    console.print(f"[bold green]✔ Sucesso! Tudo salvo em: {output_dir}/[/bold green]")