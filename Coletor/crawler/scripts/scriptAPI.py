import pandas as pd
import csv

import time
import threading
import requests
import os

from scripts.globalState import GlobalState
from scripts.console import log

from config import config

URL_API = config["api_endpoint"]

# Recebe uma string com o caminho para o arquivo e retorna o número de linhas do arquivo
def get_csv_size(path):
  count = 0
  with open(path, 'r') as file:
    for line in file:
      count += 1
  return count

def get_atual_date():
  path = 'files/atual_date.csv'
  with open(path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      year, month, day = map(int, row)
      date_dict = {'year': year, 'month': month, 'day': day}
      return date_dict

def get_data():
  processed_videos_path = "files/processed_videos.csv"
  comments_info_path = "files/comments_info.csv"
  channels_info_path = "files/channels_info.csv"
  videos_info_path = "files/videos_info.csv"

  data = {
    "processed_videos": get_csv_size(processed_videos_path) if os.path.exists(processed_videos_path) else 0,
    "comments_info": get_csv_size(comments_info_path) if os.path.exists(comments_info_path) else 0,
    "channels_info": get_csv_size(channels_info_path) if os.path.exists(channels_info_path) else 0,
    "videos_info": get_csv_size(videos_info_path) if os.path.exists(videos_info_path) else 0,
    "atual_date": get_atual_date(),
    "global_state": GlobalState.get_instance().get_state()
  }

  return data

def sendStatus():
  while True:
    data = get_data()

    response = None
    while response == None or response.status_code != 200:
      try:
        response = requests.patch(URL_API, json=data)
      except Exception as e:
        print("Unexpected in STATUS API request, trying again in ", config["try_again_timeout"], " seconds")
        time.sleep(config["try_again_timeout"])
    
    print("> Servidor de status atualizado")
    time.sleep(config["api_cooldown"])

def connectCheckAPI():
  if(URL_API != ""):
    check_threading = threading.Thread(target=sendStatus)
    check_threading.start()

