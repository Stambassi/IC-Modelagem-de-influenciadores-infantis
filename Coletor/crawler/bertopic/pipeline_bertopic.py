import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path

from visualizar_topicos import visualizar_bertopic
from otimizar_bertopic import otimizar_BERTopic, salvar_BERTopic
from preparar_dados import get_dados

from InquirerPy import prompt

import spacy

console = Console()

YOUTUBER = 'Julia MineGirl'

# Lista expandida para contexto de Youtuber Infantil/Gamer
# Remove verbos de ação genéricos e pedidos de engajamento que poluem tópicos
custom_sWords = {
    "aqui", "pra", "velho", "né", "tá", "mano", "ah",
    "dela", "ju", "beleza", "jú", "julia", "olá", "tô",
    "gente", "ta", "olha", "pá", "vi", "ai", "júlia", "será",
    "pessoal", "galerinha", "acho", "vou", "daí", "porta",
    "hein", "bora", "aham", "juma", "tipo", "então", "assim",
    "vai", "bom", "agora", "fazer", "coisa", "ver", "tudo",
    "inscreva", "canal", "like", "vídeo", "video", "sininho",
    "notificação", "compartilha", "deixa", "gostei", "comenta",
    "oi", "e aí", "eae", "fala", "galera", "todos", "bem-vindos"
}

# Intervalos de parâmetros a serem ajustados
param_ranges = {
    "n_neighbors": {"type": "int", "low": 15, "high": 60},
    
    "n_components": {"type": "int", "low": 2, "high": 5},
    
    "min_dist": {"type": "float", "low": 0.0, "high": 0.5},
    
    "min_cluster_size": {"type": "int", "low": 10, "high": 50},
    
    "min_samples": {"type": "int", "low": 5, "high": 30},
    
    "min_df": {"type": "int", "low": 2, "high": 20},
    
    "max_df": {"type": "float", "low": 1.0, "high": 1.0},
    
    "ngram_range": {"type": "categorical", "choices": [(1,1), (1,2), (1,3)]},
}

def pipeline_BERTopic(param_ranges):
    documentos = get_dados()
    
    console.print(f"[bold green]Iniciando otimização com {len(documentos)} documentos...[/bold green]")
    
    # Otimiza o BERTopic com o intervalo de parâmetros e as stopwords personalizadas
    study = otimizar_BERTopic(documentos, param_ranges, stop_words=list(custom_sWords), n_trials=30)
    
    # Verifica se o estudo encontrou algo válido antes de salvar
    if study and study.best_value > -1:
        console.print(f"[bold blue]Melhor score encontrado:[/bold blue] {study.best_value}")
        console.print(f"[bold blue]Melhores parâmetros:[/bold blue] {study.best_params}")
        salvar_BERTopic(documentos, study.best_params, stop_words=list(custom_sWords))
    else:
        console.print("[bold red]Falha na otimização. Nenhum trial válido encontrado.[/bold red]")
    
def editar_parametros():
    global param_ranges

    print("\n--- Ranges atuais ---")
    for k, v in param_ranges.items():
        if v["type"] != "categorical":
            print(f"{k}: [{v['low']} , {v['high']}] ({v['type']})")
        else:
            print(f"{k}: {v['choices']}")
    print("----------------------\n")

    change_question = [
        {
            "type": "confirm",
            "name": "change",
            "message": "Deseja alterar algum parâmetro?",
            "default": False,
        }
    ]
    should_change = prompt(change_question)["change"]
    if not should_change:
        return

    # Loop para editar cada parâmetro
    for param, cfg in param_ranges.items():

        if cfg["type"] == "categorical":
            q = [
                {
                    "type": "list",
                    "name": "value",
                    "message": f"Escolher valor para {param}:",
                    "choices": cfg["choices"],
                }
            ]
            ans = prompt(q)["value"]
            param_ranges[param]["choices"] = [ans]
            continue

        q1 = [
            {
                "type": "input",
                "name": "low",
                "message": f"Novo valor mínimo para {param} (atual {cfg['low']}):",
                "validate": lambda t: t.replace('.', '', 1).isdigit(),
            },
            {
                "type": "input",
                "name": "high",
                "message": f"Novo valor máximo para {param} (atual {cfg['high']}):",
                "validate": lambda t: t.replace('.', '', 1).isdigit(),
            }
        ]
        ans = prompt(q1)

        if cfg["type"] == "float":
            param_ranges[param]["low"] = float(ans["low"])
            param_ranges[param]["high"] = float(ans["high"])
        else:
            param_ranges[param]["low"] = int(float(ans["low"]))
            param_ranges[param]["high"] = int(float(ans["high"]))

    print("\n✔ Parâmetros atualizados!\n")

# -----------------------------------------------------
# MENU PRINCIPAL
# -----------------------------------------------------
def main_menu():
    while True:
        menu = [
            {
                "type": "list",
                "name": "acao",
                "message": "Selecione uma opção:",
                "choices": ["Pipeline", "Visualizar", "Sair"],
            }
        ]

        choice = prompt(menu)["acao"]

        if choice == "Pipeline":
            editar_parametros()
            pipeline_BERTopic(param_ranges)

        elif choice == "Visualizar":
            visualizar_bertopic()

        elif choice == "Sair":
            print("\nSaindo...\n")
            break


if __name__ == "__main__":
    main_menu()