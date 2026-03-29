# Algoritmo que reseta o Crawler com as configurações contidas em config.py
# O estado Atual será mantido

from datetime import datetime
from config import config
import csv
import os
import pandas as pd

# Mapeamento para atualizar o dashboard corretamente
MAPA_MESES_REV = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

def reset():
    # 1. Exclui o arquivo global antigo para evitar conflitos futuros
    arquivo_global_antigo = "files/atual_date.csv"
    if os.path.exists(arquivo_global_antigo):
        try:
            os.remove(arquivo_global_antigo)
        except:
            pass

    # 2. Reseta as datas nas pastas individuais de cada youtuber
    if os.path.isdir("files"):
        for youtuber_folder in os.listdir("files"):
            caminho_youtuber = os.path.join("files", youtuber_folder)
            
            if os.path.isdir(caminho_youtuber):
                caminho_csv = os.path.join(caminho_youtuber, "atual_date.csv")
                with open(caminho_csv, "w", newline="") as csvfile:
                    fieldnames = ["year", "month", "day"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        "year": config["end_date"][0],
                        "month": config["end_date"][1],
                        "day": config["end_date"][2]
                    })
    else:
        os.makedirs("files")

    # 3. Atualiza as datas no youtuberslist.csv para refletir no Dashboard
    csv_path = "youtuberslist.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        ano_reset = str(config["end_date"][0])
        mes_reset = MAPA_MESES_REV.get(config["end_date"][1], "Janeiro")

        df['ultimoAnoColetado'] = ano_reset
        df['ultimoMesColetado'] = mes_reset
        
        df.to_csv(csv_path, index=False)

    return config["end_date"][0], config["end_date"][1], config["end_date"][2]