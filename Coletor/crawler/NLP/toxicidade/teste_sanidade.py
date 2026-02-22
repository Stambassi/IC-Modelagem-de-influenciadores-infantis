import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rich.console import Console
import re

from detoxify import Detoxify

console = Console()

# Configuração global
BASE_FOLDER = Path('files')   
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_ALVO = 'toxicity'
PASTA_SAIDA_GRAFICOS = Path("files/SVT/sanidade")

# Mapeia cada youtuber para sua categoria principal (Minecraft, Roblox, etc.)
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
    Converte uma string de duração no formato ISO 8601 (ex: PT8M7S, PT1H2M10S) para segundos
    
    @param duracao_iso - String de duração enviada pela API do YouTube.
    @return int - Total de segundos convertido.
'''
def converter_duracao_iso_para_segundos(duracao_iso: str) -> int:
    # Regex para capturar grupos opcionais de Horas (H), Minutos (M) e Segundos (S)
    padrao = re.compile(r'PT(?:(?P<horas>\d+)H)?(?:(?P<minutos>\d+)M)?(?:(?P<segundos>\d+)S)?')
    match = padrao.match(duracao_iso)
    
    if not match:
        return 0

    # Extrai os valores dos grupos, tratando como 0 se não existirem
    horas = int(match.group('horas') or 0)
    minutos = int(match.group('minutos') or 0)
    segundos = int(match.group('segundos') or 0)

    # Cálculo do total de segundos
    total_segundos = (horas * 3600) + (minutos * 60) + segundos
    
    return total_segundos


'''
    Função para identificar e retornar o vídeo com maior variabilidade de toxicidade,
    ou um vídeo em específico, caso um ID seja fornecido diretamente
    
    @param lista_youtubers - Lista de nomes para varredura de arquivos
    @param metrica - "variancia" (padrão) ou "entropia" para critério de seleção (usado se video_id for None)
    @param video_id - ID específico do vídeo alvo (opcional). Ex: "dQw4w9WgXcQ"
    @return dict - Dados para o gráfico: {'id', 'youtuber', 'duracao', 'textos', 'scores'}
'''
def buscar_video_exemplo(lista_youtubers: list, metrica: str = 'variancia', video_id: str = None) -> dict:
    video_candidato = None
    maior_valor_metrica = -1.0

    # Feedback de console ajustado conforme o modo de operação
    if video_id:
        console.print(f"[bold]Buscando vídeo específico por ID: {video_id}...[/bold]")
    else:
        console.print(f"[bold]Buscando vídeo de exemplo baseado na métrica: {metrica}...[/bold]")

    for youtuber in lista_youtubers:
        path = BASE_FOLDER / youtuber
        if not path.is_dir(): continue
        
        # rglob para encontrar todas as ocorrências de tiras_video.csv nas subpastas de vídeos
        for file in path.rglob(INPUT_FILENAME):
            try:
                # Se um ID foi passado, filtra diretamente pelo nome do diretório pai
                if video_id and f"[{video_id}]" not in file.parent.name:
                    continue

                # Carregar dataframe para validar existência de dados
                df = pd.read_csv(file)

                # Ignorar vídeos muito curtos ou sem as colunas necessárias para o gráfico
                if COLUNA_ALVO not in df.columns or 'tiras' not in df.columns or len(df) < 3: 
                    continue
                
                # Alinhamento e extração de textos e scores (removendo linhas com falha de processamento)
                df_valid = df.dropna(subset=[COLUNA_ALVO, 'tiras'])
                scores = df_valid[COLUNA_ALVO].values.tolist()
                textos = df_valid['tiras'].values.tolist()
                
                valor_calculado = 0.0

                # LÓGICA 1: Retorno imediato se estiver buscando por um ID específico
                if video_id:
                    duracao = -1
                    info_path = file.parent / 'videos_info.csv'
                    if info_path.exists():
                        df_info = pd.read_csv(info_path)
                        duracao = converter_duracao_iso_para_segundos(df_info['duration'].iloc[0])
                        
                    video_candidato = {
                        'id': video_id,
                        'youtuber': youtuber,
                        'duracao': duracao,
                        'textos': textos,
                        'scores': scores
                    }
                    console.print(f"     [green]Vídeo específico identificado:[/green] {video_candidato['id']} ({video_candidato['youtuber']})")
                    return video_candidato

                # Cálculo baseado na métrica estatística escolhida para busca cega
                if metrica == "variancia":
                    valor_calculado = np.var(scores)
                elif metrica == "entropia":
                    counts = np.histogram(scores, bins=10, range=(0, 1))[0]
                    probs = counts / np.sum(counts)
                    probs = probs[probs > 0] 
                    valor_calculado = -np.sum(probs * np.log2(probs))

                # LÓGICA 2: Atualizar candidato se o valor atual superar o recorde anterior
                if valor_calculado > maior_valor_metrica:
                    maior_valor_metrica = valor_calculado
                    
                    # Recuperar ID do vídeo e duração do arquivo de metadados adjacente
                    video_id_encontrado = "Desconhecido"
                    duracao = -1
                    info_path = file.parent / 'videos_info.csv'
                    if info_path.exists():
                        df_info = pd.read_csv(info_path)
                        video_id_encontrado = df_info['video_id'].iloc[0]
                        duracao = converter_duracao_iso_para_segundos(df_info['duration'].iloc[0])

                    # Estrutura de retorno enxuta focada no gráfico
                    video_candidato = {
                        'id': video_id_encontrado,
                        'youtuber': youtuber,
                        'duracao': duracao,
                        'textos': textos,
                        'scores': scores
                    }
            except Exception as e:
                # Silenciar erros de leitura individual para não interromper a busca global
                continue

    # Feedback final
    if video_candidato and not video_id:
        console.print(f"     [green]Vídeo identificado ({metrica}):[/green] {video_candidato['id']} ({video_candidato['youtuber']})")
    elif video_id and not video_candidato:
        console.print(f"     [red]Vídeo com ID '{video_id}' não encontrado nos dados locais.[/red]")
    
    return video_candidato

'''
    Função para concatenar os trechos de texto iterativamente e rodar o Detoxify
    Testa o limite de tokens do modelo (512 tokens para arquiteturas BERT)
    
    @param textos - Lista contendo os textos (SVT) transcritos do vídeo
    @return list - Lista com os scores de toxicidade acumulados a cada nova tira adicionada
'''
def calcular_scores_acumulados(textos: list) -> list:
    console.print("[cyan]Carregando modelo Detoxify (multilingual)...[/cyan]")
    modelo_detoxify = Detoxify('multilingual')
    
    scores_acumulados = []
    texto_acumulado = ""

    for i, texto in enumerate(textos):
        # Concatena o texto atual ao montante anterior
        texto_acumulado += (" " + str(texto)) if i > 0 else str(texto)
        
        console.print(f"  -> Inferindo toxicidade para janela acumulada 0 a {i+1}...")
        
        try:
            # O Detoxify retorna um dicionário, extraímos apenas a toxicidade geral
            resultado = modelo_detoxify.predict(texto_acumulado)
            score = resultado['toxicity']
            scores_acumulados.append(score)
            
        except Exception as e:
            # Captura o erro de limite de tokens (tamanho de tensor incompatível)
            console.print(f"     [bold red]Erro de inferência no nível {i+1} (provável limite de tokens):[/bold red] {e}")
            # Adiciona -1.0 para sinalizar graficamente onde o modelo quebrou
            scores_acumulados.append(-1.0) 

    return scores_acumulados

'''
    Função para desenhar o gráfico cascata (escadinha) comparando SVTs isolados com agrupados
    
    @param video_info - Dicionário retornado pela função 'buscar_video_exemplo'
    @param scores_acumulados - Lista de scores gerados pela função 'calcular_scores_acumulados'
'''
def plotar_grafico_cascata(video_info: dict, scores_acumulados: list):
    scores_individuais = video_info['scores']
    video_id = video_info['id']
    youtuber = video_info['youtuber']
    n = len(scores_individuais)

    # Função interna para definir as cores baseadas na discretização (NT, GZ, T)
    def definir_cor(score):
        if score < 0.0: return "#34495e"   # Cinza (Erro de Tokenização)
        if score < 0.20: return "#2ecc71"  # Verde (Non-Toxic)
        if score < 0.80: return "#f1c40f"  # Amarelo (Grey-Zone)
        return "#e74c3c"                   # Vermelho (Toxic)

    # Ajuste dinâmico do tamanho da figura baseado na quantidade de tiras
    largura_fig = max(10, n * 1.2)
    altura_fig = max(6, n * 1.5)
    fig, ax = plt.subplots(figsize=(largura_fig, altura_fig))
    
    ax.set_xlim(0, n + 1)
    ax.set_ylim(- (n * 1.5) - 1, 1)
    ax.axis('off') # Remove os eixos padrão para desenhar os eixos customizados

    # Desenhar Eixo X e Y principais
    ax.annotate('', xy=(0, 0), xytext=(0, - (n * 1.5) - 0.5), arrowprops=dict(arrowstyle="->", lw=1.5)) # Y
    ax.annotate('', xy=(n + 0.8, - (n * 1.5) - 0.5), xytext=(0, - (n * 1.5) - 0.5), arrowprops=dict(arrowstyle="->", lw=1.5)) # X

    # Marcações (ticks) numéricas no Eixo X
    for i in range(1, n + 1):
        x_pos = i
        y_pos = - (n * 1.5) - 0.5
        ax.plot([x_pos, x_pos], [y_pos + 0.1, y_pos - 0.1], color='black', lw=1.5)
        ax.text(x_pos, y_pos - 0.4, str(i), ha='center', va='center', fontsize=10)

    # Loop para desenhar cada linha da cascata de cima para baixo
    for i in range(n):
        y_base = - (i * 1.5) - 1
        largura_bloco_total = i + 1
        
        # Isolar os scores individuais pertinentes à linha atual para calcular a média
        scores_janela_atual = scores_individuais[:largura_bloco_total]
        media_atual = sum(scores_janela_atual) / len(scores_janela_atual)
        
        # 1. Desenhar a barra superior (Score Acumulado)
        score_acumulado_atual = scores_acumulados[i]
        
        # Formatação do texto do bloco superior
        if score_acumulado_atual < 0:
            texto_acumulado = "ERRO LIMITE TOKENS"
        else:
            # Controle de tamanho e coerência: omite o texto "(Média)" na primeira tira
            if i == 0:
                texto_acumulado = f"{score_acumulado_atual:.2f}"
            else:
                texto_acumulado = f"{score_acumulado_atual:.2f} (Média: {media_atual:.2f})"
        
        # Renderização do retângulo superior
        rect_global = patches.Rectangle((0.5, y_base), largura_bloco_total, 0.4, fill=True, facecolor='#f5f5f5', edgecolor='black', lw=1.5)
        ax.add_patch(rect_global)
        ax.text(0.5 + largura_bloco_total / 2, y_base + 0.2, texto_acumulado, ha='center', va='center', fontsize=11, fontweight='bold')

        # 2. Desenhar as barras inferiores menores (Scores Individuais dos SVTs)
        for j in range(largura_bloco_total):
            score_ind = scores_individuais[j]
            cor_fundo = definir_cor(score_ind)
            
            # Renderização do retângulo inferior (SVT individual)
            rect_ind = patches.Rectangle((0.5 + j, y_base - 0.4), 1, 0.4, fill=True, facecolor=cor_fundo, edgecolor='black', lw=1.5)
            ax.add_patch(rect_ind)
            ax.text(0.5 + j + 0.5, y_base - 0.4 + 0.2, f"{score_ind:.2f}", ha='center', va='center', fontsize=10)

    # Metadados do Gráfico
    plt.title(f"Teste de Sanidade de Composicionalidade - Detoxify\nVídeo: {video_id} ({youtuber})", fontsize=14, pad=20)
    
    # Criar diretório e salvar
    PASTA_SAIDA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_salvamento = PASTA_SAIDA_GRAFICOS / f"cascata_sanidade_{video_id}.png"
    plt.savefig(caminho_salvamento, bbox_inches='tight', dpi=300)
    plt.close()
    
    console.print(f"[bold green]Gráfico salvo com sucesso em:[/bold green] {caminho_salvamento}")

'''
    Função orquestradora para o teste de sanidade
    
    @param lista_youtubers - Lista de nomes para varredura de arquivos
    @param metrica - "variancia" (padrão) ou "entropia" para critério de seleção (usado se video_id for None)
    @param video_id - ID específico do vídeo alvo (opcional). Ex: "dQw4w9WgXcQ"
'''
def executar_teste_sanidade(lista_youtubers: list, metrica: str ='variancia', video_id: str = None):
    console.print("[bold magenta]===== INICIANDO TESTE DE SANIDADE DO DETOXIFY =====[/bold magenta]")
    
    video_info = buscar_video_exemplo(lista_youtubers, metrica=metrica, video_id=video_id)
    
    if not video_info:
        console.print("[red]Processo abortado. Nenhum vídeo atendeu aos critérios.[/red]")
        return
        
    scores_acumulados = calcular_scores_acumulados(video_info['textos'])

    plotar_grafico_cascata(video_info, scores_acumulados)
    
    console.print("[bold magenta]===== TESTE FINALIZADO =====[/bold magenta]")

if __name__ == '__main__':
    # Definir variáveis iniciais
    lista_youtubers = list(MAPA_YOUTUBERS_CATEGORIA.keys())

    metricas = ['variancia', 'entropia']

    video_ids = ['V73bQvJE2yo', 'Xv8P3TZVcN8']

    # Executar teste
    for metrica in metricas:
        executar_teste_sanidade(lista_youtubers, metrica=metrica)

    for video_id in video_ids:
        executar_teste_sanidade(lista_youtubers, video_id=video_id)