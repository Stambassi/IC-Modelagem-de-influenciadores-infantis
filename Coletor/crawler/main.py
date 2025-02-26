from InquirerPy import prompt
import pandas as pd
from googleapiclient.discovery import build
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
import script
import analise
import video_process

console = Console()
# with open('../../../README.md',encoding="utf-8") as f:
#     md = Markdown(f.read())
#     console.print(md)
#     print("")

console.print("\n==============================================================================================",style="bold white")
console.print(" Coletor de dados do [bold red]Youtube[/] - [bold white]v1.1[/]\n Authors: João Pedro Torres, Augusto Stambassi Duarte, Lucas Carneiro Nassau Malta",style="bold white")
console.print("==============================================================================================\n",style="bold white")

csv_path = "youtuberslist.csv"
perguntas = [
    {
        "type": "list",
        "message": "Escolha uma ação",
        "choices": ["Mostrar Lista de influenciadores pesquisados", "Adicionar novo(s) influenciadore(s)", 
        "Começar a coleta de dados", "Analisar dados","Gerar speech-to-text de todos os videos",
        "Gerar speech-to-text de apenas um youtuber","Sair"]
    },
    {
        "type": "list",
        "message": "Deseja adicionar novo(s)?",
        "choices": ["Sim", "Não"]
    },
    {
        "type": "list",
        "message": "Qual tamanho do modelo do Whisper?",
        "choices": ["tiny", "base", "small", "medium", "large","turbo"]
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
            dict = {'nome': nomes_correto, 'channel_id': total_id, 'subscribers': total_subs, 'ultimoAnoColetado': "2019", 'ultimoMesColetado': "Janeiro"
            , "videosColetados":0,"comentariosColetados":0,"videosTranscritos":0}
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

def gerar_pergunta_youtube():
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            prompt (perguntas[1])
        else:
            youtubers = []
            for _, row in df.iterrows():
                youtubers.append(str(row["nome"]))
            return [{ "type": "list", "message": "Escolha um youtuber", "choices": youtubers }]

    except FileNotFoundError:
        console.print("Lista [red]vazia[/]")
        adicionar = prompt (perguntas[1])
        print(" ")
        if adicionar[0] == "Sim":
            adicionar_influenciador()

def mostrar_lista_influenciadores():
    try:
        df = pd.read_csv(csv_path)
        # print(">> Lista de canais")

        if df.empty:
            prompt (perguntas[1])
        else:
            tabela = Table(title="Influenciadores")
            tabela.add_column("Nome", justify="center", style="red")
            # tabela.add_column("Channel ID", justify="center", style="white")
            tabela.add_column("Subscribers", justify="right", style="green")
            tabela.add_column("Ultima Data de coleta", justify="center")
            tabela.add_column("Videos coletados", justify="right", style="green")
            tabela.add_column("Comentários coletados", justify="right", style="cyan")
            tabela.add_column("Videos transcritos", justify="right", style="green")
            total_videos = 0
            total_comentarios = 0
            for _, row in df.iterrows():
                mes = str(row['ultimoMesColetado'])
                ano = str(row['ultimoAnoColetado'])
                videosTranscritos = video_process.atualizar_video_total_transcritos(str(row["nome"]))
                tabela.add_row(
                    str(row["nome"]), 
                    str(row["subscribers"]), 
                    str(mes +"/"+ ano),
                    str(row["videosColetados"]),
                    str(row["comentariosColetados"]),
                    str(videosTranscritos)
                )
                total_videos += row["videosColetados"]
                total_comentarios += row["comentariosColetados"]
            console.print(tabela)
            console.print("[bold]Total de vídeos coletados: [green]"+str(total_videos)+"[/]\nTotal de comentários coletados: [cyan]"+str(total_comentarios)+"[/][/]")
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
        console.print(">> [green]Coletando dados[/]")
        script.main()
        print(" ")
    elif resultado[0] == "Analisar dados":
        console.print(">> [green]Analisando dados[/]")
        analise.main()
        print(" ")
    elif resultado[0] == "Gerar speech-to-text de todos os videos":
        modelo = prompt(perguntas[2])
        console.print(">> [green]Gerando speech-to-text[/]")
        video_process.process_all_videos(modelo)
        print(" ")
    elif resultado[0] == "Gerar speech-to-text de apenas um youtuber":
        modelo = prompt(perguntas[2])
        pergunta_youtuber = gerar_pergunta_youtube()
        print(" ")
        youtuber = prompt(pergunta_youtuber)
        print(youtuber)
        console.print(">> [green]Gerando speech-to-text do "+youtuber[0])
        video_process.process_youtuber_video(modelo,youtuber[0])
        print(" ")

    resultado = prompt (perguntas[0])
    print(" ")





