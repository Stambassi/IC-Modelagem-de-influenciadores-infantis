import pandas as pd
import re
import os
import json
from rich.console import Console
from pathlib import Path
import spacy
from nltk.stem import RSLPStemmer

console = Console()

# Inicializa o Stemmer globalmente para não recarregar no loop
stemmer = RSLPStemmer()

# Carrega lista base para validação de existência de pastas
df_raw = pd.read_csv('youtuberslist.csv')

df_raw = df_raw[df_raw['videosTranscritos'] != 0]

YOUTUBER_LIST_VALIDOS = set(df_raw['nome'].tolist())

# Regex para remover sufixos comuns de diminutivo (inho, inha, zinho, zinha)
# O '$' garante que só remove se for no final da palavra
REGEX_DIMINUTIVO = re.compile(r'(zinhos|zinhas|zinhos|zinha|zinho|inhos|inhas|inho|inha)$')

# Lista de interjeições fixas
INTERJEICOES_FIXAS = {
    "uhul", "uhuu", "oba", "vixe", "ai", "haha", "uhuuu", "wow", 
    "uiuu", "uuuu", "uau", "hey", "huh", "eeeeh", "uiiiii", 
    "ihihi", "hehehehehehe", "tum", "aaaa", "hehehehe", 
    "lalalalalala", "nha", "piuí", "uuuuh", "mamamama", 
    "nããão", "miuuuu", "buh", "brrrr", "bláááá", "prrrrrr", 
    "ihihihi", "turuturu", "ahahahahaha", "aaaai", "uuuuuu",
    "ops", "upa", "epa", "olha", "nossa", "eita", "uai", "aff"
}

# Padrões Regex para capturar o que não está na lista
    # Explicação dos Grupos:
    # 1. r'?:^u+h+u+l*$'           -> Captura variações de uhu, uhuu, uhuuuul
    # 2. r'?:^(?:k|r){3,}$'          -> Captura risadas de texto (kkkk, rrrr)
    # 3. r'?:^(?:ha|he|hi|hua|hue){2,}$' -> Captura risadas silábicas (hahaha, huehue)
    # 4. r'?:^([aeiouáéíóúãõâêôà])\1{2,}$'    -> Captura vogais únicas repetidas (aaaa, oooo, iiii)
    # 5. r'(?:^[aeiouáéíóúãõâêôà]i{2,}$)'     -> [NOVO] Captura vogal seguida de múltiplos 'i' (uiii, oiii, eiii, aiii)

REGEX_RUIDO = re.compile(
    r'(?:^u+h+u+l*$)|'    
    r'(?:^(?:k|r){3,}$)|'    
    r'(?:^(?:ha|he|hi|hua|hue){2,}$)|'    
    r'(?:^([aeiouáéíóúãõâêôà])\1{2,}$)|'    
    r'(?:^[aeiouáéíóúãõâêôà]i{2,}$)',     
    re.IGNORECASE
)

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
    'Robin Hood Gamer': 'Minecraft',
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
    Função para identificar se uma palavra é uma interjeição
    @param palavra - Palavra a ser testada
    @return True, se interjeição. False, caso contrário
'''
def eh_interjeicao(palavra):
    palavra = palavra.lower().strip()
    
    # Checagem 1: Está na lista fixa?
    if palavra in INTERJEICOES_FIXAS:
        return True
        
    # Checagem 2: Bate com os padrões de repetição?
    if REGEX_RUIDO.match(palavra):
        return True
        
    return False

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
    Função para pegar o lema do Spacy e remover o diminutivo manualmente
    - Resolve Plural: 'Gatos' -> 'Gato'
    - Remove diminutivo manualmente: 'Gatinho' -> 'Gato'

    @param token - Token a ser normalizado
    @return palavra - Token normalizado
'''
def normalizar_palavra(token):
    palavra = token.lemma_.lower().strip()
    
    # Se a palavra terminar com sufixos de diminutivo, corta
    # Ex: 'gatinho' -> 'gat'
    # Problema: 'gat' é feio. Vamos tentar ser cirúrgicos.
    
    # Estratégia Cirúrgica:
    # Se remover o sufixo deixar a palavra muito curta (<3 letras), mantém.
    # Ex: "vinho" -> "v" (Errado). "gatinho" -> "gat" (Aceitável, mas feio).
    
    # MELHOR ABORDAGEM PARA VISUALIZAÇÃO:
    # Apenas removemos o sufixo se a palavra for longa.
    # Mas para o seu caso, a lematização do Spacy já ajuda muito.
    # Vamos aplicar um corte suave.
    
    match = REGEX_DIMINUTIVO.search(palavra)
    if match:
        # Remove o sufixo encontrado
        raiz = REGEX_DIMINUTIVO.sub('', palavra)
        
        # Correção ortográfica básica pós-corte (Heurística)
        # Ex: "casinha" -> "cas" -> vira "casa"? Difícil automatizar perfeito.
        # Ex: "gatinho" -> "gat" -> vira "gato"?
        
        # Para fins de Topic Modeling, agrupar "gat" é melhor que ter "gato" e "gatinho".
        # Se quiser ficar bonito no gráfico, o BERTopic tenta achar a "palavra representativa".
        if len(raiz) > 1:
            return raiz
            
    return palavra

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
    
    console.print(f"⏳ Normalizando {len(documentos)} documentos (Lematização + Remoção de Diminutivos)...")

    # Processamento em lote
    for doc in nlp.pipe(documentos, batch_size=50, n_process=1):
        tokens_uteis = []
        for token in doc:
            # Verifica se é interjeição (Lista ou Regex)
            if eh_interjeicao(str(token)):
                continue 
    
            # Filtra por classe gramatical (Substantivo ou Nome Próprio)
            # if token.pos_ in ['NOUN', 'PROPN'] and len(token.text.strip()) > 2:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text.strip()) > 2:
                
                # 1. token.lemma_ -> O Spacy já transforma "gatos" em "gato"
                # 2. normalizar_palavra -> Remove o "inho/zinho" residual
                token_normalizado = normalizar_palavra(token)
                
                if len(token_normalizado) > 2:
                    tokens_uteis.append(token_normalizado)
        
        texto_limpo = " ".join(tokens_uteis)
        
        # Validação de string vazia
        if texto_limpo.strip():
            docs_limpos.append(texto_limpo)
        
    return docs_limpos

'''
    Recupera os dados de texto baseados no grupo escolhido, estratégia de seleção e granularidade.
    
    @param grupo_analise: "Geral", nome da Categoria (ex: "Minecraft") ou nome do Youtuber
    @param modo_selecao: Estratégia de coleta ("total", "toxico", "janela")
    @param granularidade: Define o que é um documento ("video" ou "tirinha")
        - "video": Junta todas as tiras coletadas em um único texto
        - "tirinha": Cada tira (ou janela) coletada conta como um documento separado
    @param arquivo_tirinha: Nome do arquivo CSV a buscar
    @param usar_filtro_gramatical: Se True, aplica o filtro de substantivos
    @return list[str]: Lista de documentos prontos para o BERTopic
'''
def get_dados(grupo_analise="Geral", modo_selecao="total", granularidade="video", arquivo_tirinha='tiras_video.csv', usar_filtro_gramatical=True):   
    youtubers_alvo = obter_lista_youtubers(grupo_analise)
    
    # Filtra apenas youtubers que existem no CSV original (segurança)
    youtubers_processar = [y for y in youtubers_alvo if y in YOUTUBER_LIST_VALIDOS]
    
    console.print(f"[bold cyan]Coletando dados ({modo_selecao.upper()} | Por {granularidade.upper()}) para: {grupo_analise}[/bold cyan]")
    
    documento = []
    
    for youtuber in youtubers_processar:
        base_path = Path(f'files/{youtuber}')
        if not base_path.exists():
            continue
            
        # Busca recursiva pelos arquivos de tirinhas
        for tirinha_csv_path in base_path.rglob(arquivo_tirinha):
            
            tiras_coletadas = []
            
            # Seleção da estratégia de coleta
            if modo_selecao == "total":
                tiras_coletadas = coletar_tirinhas_video_totais(tirinha_csv_path)
            
            elif modo_selecao == "toxico":
                tiras_coletadas = coletar_tirinhas_video_toxicas(tirinha_csv_path)
            
            elif modo_selecao == "janela":
                tiras_coletadas = coletar_tirinhas_video_janela(tirinha_csv_path)
            
            else:
                console.print(f"[red]Modo de seleção '{modo_selecao}' inválido.[/red]")
                return []
            
            # Aplicação da granularidade
            if tiras_coletadas:
                
                if granularidade == "video":
                    # MODO MACRO: Junta todas as cenas de um vídeo em um único documento
                    texto_video_completo = " ".join(tiras_coletadas)
                    if len(texto_video_completo.strip()) > 10:
                        documento.append(texto_video_completo)
                        
                elif granularidade == "tirinha":
                    # MODO MICRO: Cada tira (ou janela fundida) é um documento independente
                    for tira in tiras_coletadas:
                        if len(tira.strip()) > 3: # Validação mínima para não pegar pontuação solta
                            documento.append(tira)
                
                else:
                    console.print(f"[red]Granularidade '{granularidade}' inválida. Use 'video' ou 'tirinha'.[/red]")
                    return []
    
    # Estatísticas básicas
    # Nota: Se granularidade for "tirinha", 'len(documento)' será o total de tiras/janelas, não de vídeos
    label_contagem = "Vídeos" if granularidade == "video" else "Tirinhas/Janelas"
    words_count = sum(len(doc.split()) for doc in documento)

    console.print(f"-> {label_contagem} processados(as): {len(documento)}")
    console.print(f"-> Total de palavras (bruto): {words_count}")

    # Aplicação do filtro gramatical
    if usar_filtro_gramatical:
        documento = filtrar_substantivos(documento)
        words_count_clean = sum(len(doc.split()) for doc in documento)
        console.print(f"-> Total de palavras (após filtro substantivos): {words_count_clean}\n")

    return documento

if __name__ == "__main__":
    # Teste rápido se rodar o script diretamente
    docs = get_dados("Minecraft", usar_filtro_gramatical=True)