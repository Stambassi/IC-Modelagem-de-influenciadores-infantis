# Algoritmo que reseta o Crawler com as configurações contidas em config.py
# O estado Atual será mantido

from datetime import datetime
from config import config
import csv
import os

def createEmptyDir() :
  # Cria o diretório "files"
  os.makedirs("files")

  # Cria o arquivo atual_date.csv contendo iniciando com a data final da coleta
  with open("files/atual_date.csv", "w", newline="") as csvfile:
    fieldnames = ["year", "month", "day"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writerow({
      "year": config["end_date"][0],
      "month": config["end_date"][1],
      "day": config["end_date"][2]
    })

if os.path.isdir("files"):
  os.rename("files", f"files {datetime.now().strftime('%Y%m%d_%H%M%S')}")
  createEmptyDir()
else:
  createEmptyDir()
