from InquirerPy import prompt
import pandas as pd
from googleapiclient.discovery import build
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
import script

console = Console()
# with open('../../../README.md',encoding="utf-8") as f:
#     md = Markdown(f.read())
#     console.print(md)
#     print("")

console.print("\n===========================================================================================",style="bold white")
console.print(" Coletor de dados do [red]Youtube[/] - [white]v1.0[/] - Authors: João Pedro Torres, Augusto Stambassi Duarte",style="bold white")
console.print("===========================================================================================\n",style="bold white")

csv_path = "youtuberslist.csv"
perguntas = [
    {
        "type": "list",
        "message": "Escolha uma ação",
        "choices": ["Mostrar Lista de influenciadores pesquisados", "Adicionar novo(s) influenciadore(s)", "Começar a coleta de dados", "Sair"]
    },
    {
        "type": "list",
        "message": "Deseja adicionar novo(s)?",
        "choices": ["Sim", "Não"]
    }
]

api_key = "AIzaSyDb4titBt1hddZ1uC1gdvtySjyWmyYW70c"

def get_channel_id_subs(channel_name):
# Criação do serviço da API do YouTube
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Busca pelo canal com o nome especificado
    search_request = youtube.search().list(
        q=channel_name,
        type='channel',
        part='snippet',
        maxResults=1
    )
    search_response = search_request.execute()

    # Verifica se foi encontrado algum resultado
    if 'items' in search_response and len(search_response['items']) > 0:
        channel_id = search_response['items'][0]['id']['channelId']

        # Obtém informações detalhadas do canal
        channel_request = youtube.channels().list(
            id=channel_id,
            part='statistics'
        )
        channel_response = channel_request.execute()

        # Extrai a quantidade de inscritos
        if 'items' in channel_response and len(channel_response['items']) > 0:
            subscriber_count = channel_response['items'][0]['statistics']['subscriberCount']
            return channel_id, subscriber_count
        else:
            return channel_id, None
    else:
        return None, None

def is_channel_valid(data):
    test = True
    try:
        df = pd.read_csv(csv_path)
        if df.empty == False:
            for _, row in df.iterrows():
                if str(row["channel_id"]) == data[0]:
                    test = False
    except FileNotFoundError:
        test = True
    return test

def adicionar_influenciador():
    console.print(">> Inseririndo canais -> Pode inserir um ou vários, quando for mais de um usar [bold cyan],[/] para dividir")
    entrada = input("Nome(s): ")
    nomes = entrada.split(',')
    total_subs = []
    total_id = []
    nomes_correto = []
    dict = {}
    i = 0
    while i < len(nomes):
        nome = nomes[i]
        console.log("Adicionando "+nome.strip()+" ("+str(i)+")")
        data = get_channel_id_subs(nome)
        test = is_channel_valid(data)
        nomes_correto.append(script.nomeCanal(data[0]))
        if test: 
            total_id.append(data[0])
            total_subs.append(data[1])
            nomes[i] = nome.strip() # Deixar nome sem espaços em branco no comeco e final
            console.print("[green]Sucesso[/] ao inserir canal")
            dict = {'nome': nomes_correto, 'channel_id': total_id, 'subscribers': total_subs, 'ultimoAnoColetado': "2019"}
            i += 1
        else:
            console.print("[red]Erro[/] ao inserir canal: Canal já existente")
            nomes.pop(i)

        # console.log("Testando",log_locals=True)

    df = pd.DataFrame(dict)
    try:
    # Verifica se o arquivo já existe
        with open(csv_path, 'r') as f:
            header = False  # Não escreve o cabeçalho se o arquivo já existe
    except FileNotFoundError:
        header = True  # Escreve o cabeçalho se o arquivo não existe
    df.to_csv(csv_path, mode='a', index=False, header=header)
    print(" ")

def mostrar_lista_influenciadores():
    try:
        df = pd.read_csv(csv_path)
        # print(">> Lista de canais")

        if df.empty:
            prompt (perguntas[1])
        else:
            tabela = Table(title="Influenciadores")
            tabela.add_column("Nome", justify="center", style="red")
            tabela.add_column("Channel ID", justify="center", style="white")
            tabela.add_column("Subscribers", justify="right", style="green")
            tabela.add_column("Ultimo Ano Coletado", justify="center")

            for _, row in df.iterrows():
                tabela.add_row(
                    str(row["nome"]), 
                    str(row["channel_id"]), 
                    str(row["subscribers"]), 
                    str(row["ultimoAnoColetado"])
                )
            console.print(tabela)
            print(" ")
    except FileNotFoundError:
        console.print("Lista [red]vazia[/]")
        adicionar = prompt (perguntas[1])
        print(" ")
        if adicionar[0] == "Sim":
            adicionar_influenciador()

    
resultado = prompt (perguntas[0])
print(" ")
while resultado[0] != "Sair":
    if resultado[0] == "Mostrar Lista de influenciadores pesquisados":
        mostrar_lista_influenciadores()
    elif resultado [0] == "Adicionar novo(s) influenciadore(s)":
        adicionar_influenciador()
    elif resultado [0] == "Começar a coleta de dados":
        print(">> Coletando")
        script.main()
        print(" ")
    resultado = prompt (perguntas[0])
    print(" ")





