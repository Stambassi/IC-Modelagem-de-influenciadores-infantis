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

custom_sWords = {"aqui","pra","velho","né","tá","mano","ah",
                 "dela","ju","beleza","jú","julia","olá","tô",
                 "gente","ta","olha","pá","vi","ai","júlia","será",
                 "pessoal","galerinha","acho", "vou", 
                 "daí", "porta","hein","bora","aham","juma"}

param_ranges = {
    "n_neighbors": {"type": "int", "low": 5, "high": 50},
    "n_components": {"type": "int", "low": 2, "high": 5},
    "min_dist": {"type": "float", "low": 0.0, "high": 0.5},
    "min_cluster_size": {"type": "int", "low": 2, "high": 20},
    "min_samples": {"type": "int", "low": 2, "high": 20},
    "min_df": {"type": "float", "low": 0.0, "high": 0.1},
    "max_df": {"type": "float", "low": 1.0, "high": 1.0},
    "ngram_range": {"type": "categorical", "choices": [(1,1), (1,2)]},
}

def pipeline_BERTopic(param_ranges):
    documentos = get_dados()
    study = otimizar_BERTopic(documentos, param_ranges,n_trials=20)
    salvar_BERTopic(documentos, study.best_params)
    
def editar_parametros():
    global param_ranges

    print("\n--- Ranges atuais ---")
    for k, v in param_ranges.items():
        if v["type"] != "categorical":
            print(f"{k}: [{v['low']} , {v['high']}]")
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
        param_ranges[param]["low"] = float(ans["low"]) if cfg["type"] == "float" else int(ans["low"])
        param_ranges[param]["high"] = float(ans["high"]) if cfg["type"] == "float" else int(ans["high"])

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

