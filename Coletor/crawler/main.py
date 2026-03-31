from InquirerPy import prompt
from InquirerPy import inquirer
import pandas as pd
from googleapiclient.discovery import build
from rich.console import Console
from rich.table import Table
import script
from scripts.analise import analise_completa
from scripts.reset import reset
from pathlib import Path
import video_process
import os
import csv

console = Console()

console.print("\n==============================================================================================",style="bold white")
console.print(" Coletor de dados do [bold red]Youtube[/] - [bold white]v1.1[/]\n Authors: João Pedro Torres, Augusto Stambassi Duarte, Lucas Carneiro Nassau Malta",style="bold white")
console.print("==============================================================================================\n",style="bold white")

csv_path = "youtuberslist.csv"

MAPA_MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# -------------------------------------------------------------------
# CONFIGURAÇÃO DE PERGUNTAS (INQUIRER)
# -------------------------------------------------------------------
perguntas = {
    "menu_principal": {
        "type": "list",
        "message": "Escolha uma ação",
        "choices": ["Mostrar Lista de influenciadores pesquisados", "Adicionar novo(s) influenciadore(s)", 
        "Coletar dados", "Analisar dados","Transcrever vídeos","Baixar áudios","Reiniciar data de coleta","Sair"]
    },
    "whisper_model": {
        "type": "list",
        "message": "Qual tamanho do modelo do Whisper?",
        "choices": ["tiny", "base", "small", "medium", "large", "turbo"]
    },
    "coleta_modo": {
        "type": "list",
        "message": "Coletar dados >> Coletar de todos os youtubers ou escolher alguns?",
        "choices": ["Todos", "Escolher", "Voltar"]
    },
    "alvo_modo": {
        "type": "list",
        "message": "Todos os youtubers ou apenas um?",
        "choices": ["Todos", "Escolher", "Voltar"]
    },
    "transcrever_download": {
        "type": "list",
        "message": "Transcrever vídeo >> Precisa fazer o download do áudio se não existir?",
        "choices": ["Sim", "Não (Apenas locais)"]
    }
}

api_key = "AIzaSyDb4titBt1hddZ1uC1gdvtySjyWmyYW70c"

# -------------------------------------------------------------------
# FUNÇÕES DE APOIO (YOUTUBE API & VALIDATION)
# -------------------------------------------------------------------

def get_channel_id_subs(channel_name):
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_request = youtube.search().list(q=channel_name, type='channel', part='snippet', maxResults=1)
    search_response = search_request.execute()

    if 'items' in search_response and len(search_response['items']) > 0:
        channel_id = search_response['items'][0]['id']['channelId']
        channel_request = youtube.channels().list(id=channel_id, part='statistics')
        channel_response = channel_request.execute()

        if 'items' in channel_response and len(channel_response['items']) > 0:
            subscriber_count = channel_response['items'][0]['statistics']['subscriberCount']
            return channel_id, subscriber_count
    return None, None

def is_channel_valid(channel_id):
    if not os.path.exists(csv_path): return True
    df = pd.read_csv(csv_path)
    return not (df['channel_id'] == channel_id).any()

def adicionar_influenciador():
    console.print(">> Inserindo canais -> Use [bold cyan],[/] para múltiplos")
    entrada = input("Nome(s): ")
    nomes = [n.strip() for n in entrada.split(',')]
    
    novos_dados = []
    for nome in nomes:
        console.log(f"Buscando {nome}...")
        c_id, subs = get_channel_id_subs(nome)
        
        if c_id and is_channel_valid(c_id):
            nome_oficial = script.nomeCanal(c_id)
            novos_dados.append({
                'nome': nome_oficial, 'channel_id': c_id, 'subscribers': subs,
                'ultimoAnoColetado': "2020", 'ultimoMesColetado': "Janeiro",
                'videosColetados': 0, 'comentariosColetados': 0, 'videosTranscritos': 0
            })
            console.print(f"[green]Sucesso[/] ao inserir {nome_oficial}")
        else:
            console.print(f"[red]Erro[/] ao inserir {nome}: Canal já existe ou não encontrado")

    if novos_dados:
        df_novos = pd.DataFrame(novos_dados)
        header = not os.path.exists(csv_path)
        df_novos.to_csv(csv_path, mode='a', index=False, header=header)

def obter_youtubers_csv():
    if not os.path.exists(csv_path): return []
    return pd.read_csv(csv_path)['nome'].tolist()

def atualizar_lista_influenciadores():
    csv_path = "youtuberslist.csv"
    dir_data = Path("data")
    
    if not os.path.exists(csv_path):
        console.print("[red]Erro: youtuberslist.csv não encontrado.[/red]")
        return
    
    df_dashboard = pd.read_csv(csv_path)
    
    console.print("[bold blue]Sincronizando Dashboard com a pasta 'data' (Parquet)...[/bold blue]")

    for index, row in df_dashboard.iterrows():
        youtuber = row['nome']
        
        # Caminhos dos arquivos de índice e de data
        path_index = dir_data / f"{youtuber}_index.parquet"
        path_atual_date = dir_data / youtuber / "atual_date.csv"
        
        # 1. Atualizar Contagens (Vídeos, Comentários, Transcrições) via Parquet
        if path_index.exists():
            try:
                df_idx = pd.read_parquet(path_index)
                
                # Total de vídeos mapeados no índice
                total_videos = len(df_idx)
                
                # Soma de comentários (baseado na coluna num_comments do índice)
                # Se a coluna não existir, usa 0
                total_comments = int(df_idx['num_comments'].sum()) if 'num_comments' in df_idx.columns else 0
                
                # Total de transcritos (onde has_transcript é True)
                total_transcritos = int(df_idx['has_transcript'].sum()) if 'has_transcript' in df_idx.columns else 0
                
                df_dashboard.at[index, 'videosColetados'] = total_videos
                df_dashboard.at[index, 'comentariosColetados'] = total_comments
                df_dashboard.at[index, 'videosTranscritos'] = total_transcritos
                
            except Exception as e:
                console.print(f"[red]Erro ao ler índice de {youtuber}: {e}[/red]")
        
        # 2. Atualizar Datas de Progresso via atual_date.csv individual
        if path_atual_date.exists():
            try:
                # O arquivo atual_date não tem header, então lê os valores diretamente
                df_date = pd.read_csv(path_atual_date)
                if not df_date.empty:
                    ano = df_date.iloc[0]['year']
                    mes_num = df_date.iloc[0]['month']
                    
                    df_dashboard.at[index, 'ultimoAnoColetado'] = str(ano)
                    df_dashboard.at[index, 'ultimoMesColetado'] = MAPA_MESES.get(mes_num, "Janeiro")
            except Exception as e:
                console.print(f"[yellow]Aviso: Falha ao sincronizar data de {youtuber}: {e}[/yellow]")

    # Salva o resultado final no CSV do Dashboard
    df_dashboard.to_csv(csv_path, index=False)
    console.print("[bold green]✔ Dashboard reconstruído e sincronizado com sucesso![/bold green]")

def mostrar_lista_influenciadores():
    if not os.path.exists(csv_path):
        console.print("[red]Lista vazia![/]")
        return

    df = pd.read_csv(csv_path)
    tabela = Table(title="Influenciadores")
    tabela.add_column("Nome", justify="center", style="red")
    tabela.add_column("Subscribers", justify="right", style="green")
    tabela.add_column("Última Coleta", justify="center")
    tabela.add_column("Vídeos", justify="right", style="green")
    tabela.add_column("Comentários", justify="right", style="cyan")
    tabela.add_column("Transcritos", justify="right", style="magenta")

    for _, row in df.iterrows():
        tabela.add_row(
            str(row["nome"]), str(row["subscribers"]),
            f"{row['ultimoMesColetado']}/{row['ultimoAnoColetado']}",
            str(row["videosColetados"]), str(row["comentariosColetados"]),
            str(row["videosTranscritos"])
        )
    
    console.print(tabela)
    console.print(f"Total Vídeos: [green]{df['videosColetados'].sum()}[/] | "
                  f"Comentários: [cyan]{df['comentariosColetados'].sum()}[/] | "
                  f"Transcritos: [magenta]{df['videosTranscritos'].sum()}[/]")

# -------------------------------------------------------------------
# LOOP PRINCIPAL
# -------------------------------------------------------------------

def main():
    while True:
        acao = inquirer.select(
            message=perguntas["menu_principal"]["message"],
            choices=perguntas["menu_principal"]["choices"]
        ).execute()

        if acao == "Sair": break

        if acao == "Mostrar Lista de influenciadores pesquisados":
            atualizar_lista_influenciadores()
            mostrar_lista_influenciadores()

        elif acao == "Adicionar novo(s) influenciadore(s)":
            adicionar_influenciador()

        elif acao == "Coletar dados":
            modo = inquirer.select(message=perguntas["coleta_modo"]["message"], choices=perguntas["coleta_modo"]["choices"]).execute()
            if modo == "Todos":
                script.main()
            elif modo == "Escolher":
                lista = obter_youtubers_csv()
                selecionados = inquirer.checkbox(message="Escolha os youtubers:", choices=lista).execute()
                if selecionados: script.coletar_videos_youtuber(selecionados)

        elif acao == "Analisar dados":
            analise_completa()

        elif acao == "Transcrever vídeos":
            # 1. Pergunta sobre o download
            quer_baixar = inquirer.select(message=perguntas["transcrever_download"]["message"], choices=perguntas["transcrever_download"]["choices"]).execute()
            tipo_transcricao = "transcrever" if "Sim" in quer_baixar else "transcrever_local"
            
            # 2. Pergunta sobre o alvo
            alvo_modo = inquirer.select(message=perguntas["alvo_modo"]["message"], choices=perguntas["alvo_modo"]["choices"]).execute()
            if alvo_modo == "Voltar": continue

            # 3. Pergunta sobre o modelo Whisper
            modelo = inquirer.select(message=perguntas["whisper_model"]["message"], choices=perguntas["whisper_model"]["choices"]).execute()
            
            if alvo_modo == "Todos":
                video_process.orquestrar_processamento("Geral", tipo_transcricao, modelo)
            else:
                lista = obter_youtubers_csv()
                alvo = inquirer.select(message="Selecione o Youtuber:", choices=lista).execute()
                video_process.orquestrar_processamento(alvo, tipo_transcricao, modelo)

        elif acao == "Baixar áudios":
            alvo_modo = inquirer.select(message=perguntas["alvo_modo"]["message"], choices=perguntas["alvo_modo"]["choices"]).execute()
            if alvo_modo == "Voltar": continue
            
            if alvo_modo == "Todos":
                video_process.orquestrar_processamento("Geral", "baixar")
            else:
                lista = obter_youtubers_csv()
                alvo = inquirer.select(message="Selecione o Youtuber:", choices=lista).execute()
                video_process.orquestrar_processamento(alvo, "baixar")

        elif acao == "Reiniciar data de coleta":
            data = reset()
            console.print(f">> Data reiniciada para: {data[2]}/{data[1]}/{data[0]}")

if __name__ == "__main__":
    main()