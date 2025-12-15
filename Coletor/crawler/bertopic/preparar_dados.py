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
    Coleta TODAS as tirinhas de um vídeo, sem filtragem
    
    @param tirinha_csv_path - Caminho para o arquivo CSV das tirinhas
    @return list[str] - Lista com todas as tiras do vídeo
'''
def coletar_tirinhas_video_totais(tirinha_csv_path) -> list[str]:
    try:
        tiras_csv = pd.read_csv(tirinha_csv_path)
        return tiras_csv['tiras'].tolist()
    except Exception as e:
        console.log(f"[red]Erro na coleta total:[/] {e}")
        return []
    
'''
    Coleta APENAS as tirinhas marcadas como 'T' (Tóxicas) no arquivo de sequência
    
    @param tirinha_csv_path - Caminho para o arquivo CSV das tirinhas
    @return list[str] - Lista contendo apenas as tiras tóxicas
'''
def coletar_tirinhas_video_toxicas(tirinha_csv_path) -> list[str]:
    try:
        sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
        
        if not sequencias_path.exists():
            return []
            
        sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()
        tiras_csv = pd.read_csv(tirinha_csv_path)
        tiras_brutas = tiras_csv['tiras'].tolist()

        indices_selecionados = set()
        
        # Garante que não será acessado um índice que não existe em uma das listas
        limit = min(len(sequencia), len(tiras_brutas))
        
        for i in range(limit):
            if sequencia[i] == "T":
                indices_selecionados.add(i)

        tiras = [tiras_brutas[i] for i in sorted(indices_selecionados)]
        return tiras

    except Exception as e:
        console.log(f"[red]Erro na coleta tóxica:[/] {e}")
        return []

'''
    Recebe uma lista de tuplas (inicio, fim) e funde intervalos sobrepostos ou adjacentes
    Ex: [(10, 18), (15, 23)] -> [(10, 23)]
'''
def fundir_intervalos(intervalos):
    if not intervalos:
        return []

    # Ordena pelo início do intervalo
    intervalos.sort(key=lambda x: x[0])

    fundidos = [intervalos[0]]

    for atual_inicio, atual_fim in intervalos[1:]:
        ultimo_inicio, ultimo_fim = fundidos[-1]

        # Se o início do atual for menor ou igual ao fim do último (+1 para juntar adjacentes)
        if atual_inicio <= ultimo_fim + 1:
            # Funde: mantém o início do último e pega o maior fim entre os dois
            fundidos[-1] = (ultimo_inicio, max(ultimo_fim, atual_fim))
        else:
            # Não sobrepõe, adiciona novo intervalo
            fundidos.append((atual_inicio, atual_fim))

    return fundidos

'''
    Coleta tirinhas tóxicas com uma janela de contexto.
    Nota: A lógica de fusão de janelas (4 antes/depois) será implementada na próxima etapa.
    Por enquanto, retorna vazio ou lógica antiga para manter estrutura.
    
    @param tirinha_csv_path - Caminho para o arquivo CSV das tirinhas
    @return list[str] - Lista de tiras (Tóxicas + Contexto)
'''
def coletar_tirinhas_video_janela(tirinha_csv_path) -> list[str]:
    try:
        sequencias_path = tirinha_csv_path.parent / "sequencias" / "sequencia_toxicidade.csv"
        
        if not sequencias_path.exists():
            return []
            
        sequencia = pd.read_csv(sequencias_path, header=None)[0].tolist()
        tiras_df = pd.read_csv(tirinha_csv_path)
        tiras_brutas = tiras_df['tiras'].tolist()
        total_tiras = len(tiras_brutas)

        # Identificar Intervalos Brutos
        intervalos_brutos = []
        limit = min(len(sequencia), total_tiras)
        
        janela = 4 # Janela definida na análise estatística

        for i in range(limit):
            if sequencia[i] == "T":
                inicio = max(0, i - janela)
                fim = min(total_tiras - 1, i + janela)
                intervalos_brutos.append((inicio, fim))

        if not intervalos_brutos:
            return []

        # Funde Intervalos (Merge)
        # Transforma [[10,18], [15,23]] em [[10,23]]
        intervalos_finais = fundir_intervalos(intervalos_brutos)

        #Extrai Texto dos Intervalos Fundidos
        documentos_janela = []
        for inicio, fim in intervalos_finais:
            # Pega o slice do intervalo (fim+1 porque o slice do python é exclusivo no final)
            segmento = tiras_brutas[inicio : fim + 1]
            texto_segmento = " ".join(segmento)
            documentos_janela.append(texto_segmento)
        
        # Retorna uma lista de "cenas" tóxicas
        return documentos_janela

    except Exception as e:
        console.log(f"[red]Erro coleta janela: {e}[/red]")
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
    Recupera os dados de texto baseados no grupo escolhido e na estratégia de seleção
    
    @param grupo_analise: "Geral", nome da Categoria (ex: "Minecraft") ou nome do Youtuber
    @param modo_selecao: Estratégia de coleta ("total", "toxico", "janela")
    @param arquivo_tirinha: Nome do arquivo CSV a buscar
    @param usar_filtro_gramatical: Se True, aplica o filtro de substantivos
    @return list[str]: Lista de documentos (cada string é um vídeo processado)
'''
def get_dados(grupo_analise="Geral", modo_selecao="total", arquivo_tirinha='tiras_video.csv', usar_filtro_gramatical=True):   
    youtubers_alvo = obter_lista_youtubers(grupo_analise)
    
    # Filtra apenas youtubers que existem no CSV original (segurança)
    youtubers_processar = [y for y in youtubers_alvo if y in YOUTUBER_LIST_VALIDOS]
    
    console.print(f"[bold cyan]Coletando dados ({modo_selecao.upper()}) para o grupo: {grupo_analise}[/bold cyan]")
    
    documento = []
    
    for youtuber in youtubers_processar:
        base_path = Path(f'files/{youtuber}')
        if not base_path.exists():
            continue
            
        # Busca recursiva pelos arquivos de tirinhas
        for tirinha_csv_path in base_path.rglob(arquivo_tirinha):
            
            tiras_coletadas = []
            
            # Seleção da Estratégia
            if modo_selecao == "total":
                tiras_coletadas = coletar_tirinhas_video_totais(tirinha_csv_path)
            
            elif modo_selecao == "toxico":
                tiras_coletadas = coletar_tirinhas_video_toxicas(tirinha_csv_path)
            
            elif modo_selecao == "janela":
                tiras_coletadas = coletar_tirinhas_video_janela(tirinha_csv_path)
            
            else:
                console.print(f"[red]Modo de seleção '{modo_selecao}' inválido.[/red]")
                return []
            
            # Processamento Pós-Coleta
            if tiras_coletadas:
                # Junta todas as cenas de um vídeo em um único documento 
                # para manter a unidade de análise como "Vídeo"
                texto_video_completo = " ".join(tiras_coletadas)

                # Validação mínima de tamanho
                if len(texto_video_completo.strip()) > 10:
                    documento.append(texto_video_completo)
    
    # Estatísticas básicas
    words_count = sum(len(video.split()) for video in documento)

    console.print(f"-> Vídeos processados: {len(documento)}")
    console.print(f"-> Total de palavras (bruto): {words_count}")

    # Aplicação do Filtro Gramatical (Integrado)
    if usar_filtro_gramatical:
        # Certifique-se de que a função filtrar_substantivos está disponível
        documento = filtrar_substantivos(documento)
        words_count_clean = sum(len(video.split()) for video in documento)
        console.print(f"-> Total de palavras (após filtro substantivos): {words_count_clean}\n")

    return documento

if __name__ == "__main__":
    # Teste rápido se rodar o script diretamente
    docs = get_dados("Minecraft", usar_filtro_gramatical=True)