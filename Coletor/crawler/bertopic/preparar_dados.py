import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import spacy

console = Console()

# Carrega lista base para validação de existência de pastas
df_raw = pd.read_csv('youtuberslist.csv')

df_raw = df_raw[df_raw['videosTranscritos'] != 0]

YOUTUBER_LIST_VALIDOS = set(df_raw['nome'].tolist())

# -------------------------------------------------------------------
# CONFIGURAÇÃO DE GRUPOS E CATEGORIAS
# -------------------------------------------------------------------
MAPA_YOUTUBERS_CATEGORIA = {
    'Amy Scarlet': 'Roblox',
    'AuthenticGames': 'Minecraft',
    'Cadres': 'Minecraft',
    'Julia MineGirl': 'Roblox',
    'Kass e KR': 'Minecraft',
    'Lokis': 'Roblox',
    'Luluca Games': 'Roblox',
    'Papile': 'Roblox',
    #'Robin Hood Gamer': 'Minecraft',
    'TazerCraft': 'Minecraft',
    'Tex HS': 'Misto'
}

'''
    Função para identificar a lista de youtubers com base no filtro solicitado:
    - 'Geral': Retorna todos.
    - Nome de Categoria (ex: 'Minecraft'): Retorna membros da categoria.
    - Nome de Youtuber (ex: 'Julia MineGirl'): Retorna apenas o próprio.
'''
def obter_lista_youtubers(nome_grupo: str) -> list:
    if nome_grupo == "Geral":
        return list(MAPA_YOUTUBERS_CATEGORIA.keys())
    
    # Verifica se é uma categoria (ex: Minecraft, Roblox)
    categorias = set(MAPA_YOUTUBERS_CATEGORIA.values())
    if nome_grupo in categorias:
        return [y for y, cat in MAPA_YOUTUBERS_CATEGORIA.items() if cat == nome_grupo]
    
    # Verifica se é um youtuber individual
    if nome_grupo in MAPA_YOUTUBERS_CATEGORIA:
        return [nome_grupo]
        
    console.print(f"[bold red]Aviso:[/] Grupo ou Youtuber '{nome_grupo}' não encontrado no mapa. Retornando lista vazia.")
    return []

'''
    Função para encontrar o JSON de transcrição de um vídeo

    @param video_path - Caminho para o arquivo JSON com a transcrição do vídeo
    @return texto_limpo - JSON limpo da transcrição do vídeo
'''
def coletar_informacoes_youtuber(video_path) -> str: 
    try:
        with open(video_path, 'r') as video_data_json:
            video_data = json.load(video_data_json)
            texto_limpo = video_data['text']
    except:
        texto_limpo = ""
    return texto_limpo

'''
    Função para coletar a lista de tirinhas de um vídeo dentro de uma janela de tempo
    determinada por eventos de toxicidade
    
    @param tirinha_csv_path - Caminho para o arquivo CSV que contém as tirinhas de um vídeo
    @return list[str] - Lista de strings, em que cada string é uma tira
'''
def coletar_tirinhas_video_janela_tempo(tirinha_csv_path) -> list[str]:
    try:
        sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
        # Verifica se arquivo existe antes de abrir
        if not sequencias_path.exists():
            return ""
            
        sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()
        tiras_csv = pd.read_csv(tirinha_csv_path)
        tiras = tiras_csv['tiras'].tolist()

        indices_selecionados = set()

        for i, estado in enumerate(sequencia):
            if estado == "T":
                inicio = max(0, i - 3)
                fim = min(len(tiras) - 1, i + 3)
                for j in range(inicio, fim + 1):
                    indices_selecionados.add(j)

        tirinhas_coletadas = [tiras[i] for i in sorted(indices_selecionados)]
        return " ".join(tirinhas_coletadas)

    except Exception as e:
        console.log(f"Erro janela tempo: {e}")
        return ""

'''
    Função para coletar a lista de todas as tirinhas de um vídeo
    
    @param tirinha_csv_path - Caminho para o arquivo CSV que contém as tirinhas de um vídeo
    @return list[str] - Lista de strings, em que cada string é uma tira
'''
def coletar_tirinhas_video(tirinha_csv_path, filtro=None) -> list[str]:
    try:
        tiras_csv = pd.read_csv(tirinha_csv_path)
        
        if filtro is None:
            tiras = tiras_csv['tiras'].tolist()
        else:
            sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
            if not sequencias_path.exists():
                return []
                
            sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()
            tiras_brutas = tiras_csv['tiras'].tolist()

            indices_selecionados = set()
            limit = min(len(sequencia), len(tiras_brutas))
            
            for i in range(limit):
                if sequencia[i] == filtro:
                    indices_selecionados.add(i)

            tiras = [tiras_brutas[i] for i in sorted(indices_selecionados)]
            
        return tiras

    except Exception as e:
        console.log(f"Erro coleta simples: {e}")
        return []

'''
    Função de filtro gramatical com o Spacy
    - Mantém apenas Substantivos (NOUN) e Nomes Próprios (PROPN)
    - Remove verbos, advérbios, pronomes, etc.
'''
def filtrar_substantivos(documentos):
    # Carrega o modelo desabilitando componentes pesados desnecessários
    try:
        nlp = spacy.load("pt_core_news_lg", disable=['parser', 'ner'])
    except:
        try:
            nlp = spacy.load("pt_core_news_md", disable=['parser', 'ner'])
        except:
            console.print("[red]Erro: Nenhum modelo Spacy (lg/md) encontrado.[/red]")
            return documentos
        
    docs_limpos = []
    
    console.print(f"⏳ Aplicando filtro gramatical em {len(documentos)} documentos...")

    for doc in nlp.pipe(documentos, batch_size=50, n_process=1):
        # Filtra tokens:
        # 1. É substantivo (NOUN) ou Nome Próprio (PROPN)?
        # 2. Tem mais de 2 letras? (Evita ruídos curtos)
        tokens_uteis = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text.strip()) > 2]
        
        # Reconstrói o texto
        docs_limpos.append(" ".join(tokens_uteis))
        
    return docs_limpos

'''
    Recupera os dados de texto baseados no grupo escolhido.
    
    @param grupo_analise: "Geral", nome da Categoria (ex: "Minecraft") ou nome do Youtuber.
    @param arquivo_tirinha: Nome do arquivo CSV a buscar.
    @param usar_filtro_gramatical: Se True, aplica o filtro de substantivos.
'''
def get_dados(grupo_analise="Geral", arquivo_tirinha='tiras_video.csv', usar_filtro_gramatical=True):    
    youtubers_alvo = obter_lista_youtubers(grupo_analise)
    
    # Filtra apenas youtubers que existem no CSV original (segurança)
    youtubers_processar = [y for y in youtubers_alvo if y in YOUTUBER_LIST_VALIDOS]
    
    console.print(f"[bold cyan]Coletando dados para o grupo: {grupo_analise}[/bold cyan]")
    console.print(f"Youtubers incluídos: {youtubers_processar}")

    documento = []
    
    for youtuber in youtubers_processar:
        base_path = Path(f'files/{youtuber}')
        if not base_path.exists():
            continue
            
        # Busca recursiva pelos arquivos de tirinhas
        for tirinha_csv_path in base_path.rglob(arquivo_tirinha):
            # Opção A: Coletar tudo
            tiras_youtuber = coletar_tirinhas_video(tirinha_csv_path)
            
            # Opção B: Coletar com janela
            # tiras_youtuber = coletar_tirinhas_video_janela_tempo(tirinha_csv_path)

            # Transforma lista de tiras em uma única string (um documento por vídeo)
            if isinstance(tiras_youtuber, list):
                texto_video_completo = " ".join(tiras_youtuber)
            else:
                texto_video_completo = tiras_youtuber # Caso venha string direta da janela

            if texto_video_completo.strip():
                documento.append(texto_video_completo)
    
    # Estatísticas básicas
    words_count = sum(len(video.split()) for video in documento)

    console.print(f"-> Vídeos processados: {len(documento)}")
    console.print(f"-> Total de palavras (bruto): {words_count}")

    # Aplicação do Filtro Gramatical (Integrado)
    if usar_filtro_gramatical:
        documento = filtrar_substantivos(documento)
        words_count_clean = sum(len(video.split()) for video in documento)
        console.print(f"-> Total de palavras (após filtro substantivos): {words_count_clean}\n")

    return documento

if __name__ == "__main__":
    # Teste rápido se rodar o script diretamente
    docs = get_dados("Minecraft", usar_filtro_gramatical=True)