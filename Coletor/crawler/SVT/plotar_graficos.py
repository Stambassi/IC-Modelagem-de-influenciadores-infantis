import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rich.console import Console
import re
import math

console = Console()

# Configuração global
BASE_FOLDER = Path('files')   
INPUT_FILENAME = 'tiras_video.csv'
COLUNA_ALVO = 'toxicity'

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
    Converte uma string de duração no formato ISO 8601 (ex: PT8M7S, PT1H2M10S) para segundos.
    
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
    Função para identificar e retornar o vídeo com maior variabilidade de toxicidade
    Serve para gerar o "Estudo de Caso" que justifica a análise segmentada (SVT)
    
    @param lista_youtubers - Lista de nomes para varredura de arquivos
    @param metrica - "variancia" (padrão) ou "entropia" para critério de seleção
    @return dict - Dados do vídeo: {'id', 'youtuber', 'scores', 'media_global', 'metrica_valor', 'path'}
'''
def buscar_video_exemplo(lista_youtubers: list, metrica: str = 'variancia') -> dict:
    video_candidato = None
    maior_valor_metrica = -1.0

    console.print(f"[bold]Buscando vídeo de exemplo baseado em: {metrica}...[/bold]")

    for youtuber in lista_youtubers:
        path = BASE_FOLDER / youtuber
        if not path.is_dir(): continue
        
        # rglob para encontrar todas as ocorrências de tiras_video.csv nas subpastas de vídeos
        for file in path.rglob(INPUT_FILENAME):
            try:
                # Carregar dataframe para validar existência de dados
                df = pd.read_csv(file)

                # Ignorar vídeos muito curtos ou sem a coluna de toxicidade
                if COLUNA_ALVO not in df.columns or len(df) < 5: 
                    continue
                
                # Extração dos scores válidos
                scores = df[COLUNA_ALVO].dropna().values
                valor_calculado = 0.0

                # Cálculo baseado na métrica estatística escolhida
                if metrica == "variancia":
                    valor_calculado = np.var(scores)
                elif metrica == "entropia":
                    counts = np.histogram(scores, bins=10, range=(0, 1))[0]
                    probs = counts / np.sum(counts)
                    probs = probs[probs > 0] 
                    valor_calculado = -np.sum(probs * np.log2(probs))

                # Atualizar candidato se o valor atual superar o recorde anterior
                if valor_calculado > maior_valor_metrica:
                    maior_valor_metrica = valor_calculado
                    
                    # Recuperar ID do vídeo do arquivo de metadados adjacente
                    video_id = "Desconhecido"
                    duracao = -1
                    info_path = file.parent / 'videos_info.csv'
                    if info_path.exists():
                        df_info = pd.read_csv(info_path)
                        video_id = df_info['video_id'].iloc[0]
                        duracao = converter_duracao_iso_para_segundos(df_info['duration'].iloc[0])

                    # Estrutura de retorno com metadados completos
                    video_candidato = {
                        'id': video_id,
                        'youtuber': youtuber,
                        'duracao': duracao,
                        'scores': scores,
                        'media_global': np.mean(scores),
                        'metrica_valor': valor_calculado,
                        'path': file.parent 
                    }
            except Exception as e:
                # Silenciar erros de leitura individual para não interromper a busca global
                continue

    if video_candidato:
        console.print(f"     [green]Vídeo identificado:[/green] {video_candidato['id']} ({video_candidato['youtuber']})")
    
    return video_candidato

'''
    Função para coletar os dados de toxicidade de todas as granularidades de um vídeo específico
    
    @param video_info - Dicionário retornado pela função buscar_video_exemplo
    @param granularidades - Lista de tempos em segundos para buscar (incluindo 'global')
    @return dict - Mapeamento de granularidade para lista de scores de toxicidade
'''
def coletar_dados_granularidade_exemplo(video_info: dict, granularidades: list) -> dict:
    if not video_info:
        return {}

    # O pipeline salva as multi-granularidades na subpasta 'tiras'
    caminho_tiras = video_info['path'] / 'tiras'
    resultados_granularidade = {}

    console.print(f"[bold blue]Coletando granularidades para o vídeo: {video_info['id']}[/bold blue]")

    # Iterar sobre cada nível de tempo definido no experimento
    for g in granularidades:
        arquivo = caminho_tiras / f"tiras_video_{g}.csv"
        
        if arquivo.exists():
            try:
                # Carregar dados segmentados da escala atual
                df = pd.read_csv(arquivo)
                if COLUNA_ALVO in df.columns:
                    scores = df[COLUNA_ALVO].dropna().tolist()
                    resultados_granularidade[g] = scores
                    console.print(f"     -> Escala {g}s: {len(scores)} fatias carregadas.")
            except Exception as e:
                console.print(f"     [red]Erro ao carregar escala {g}s: {e}[/red]")
        else:
            console.print(f"     [yellow]Aviso: Escala {g}s não encontrada.[/yellow]")

    return resultados_granularidade

'''
    Função para gerar um gráfico de linhas sobrepostas comparando diferentes granularidades de extração

    @param video_info - Dicionário com metadados (id, youtuber, scores, duracao)
    @param dados_escalas - Dicionário {tempo: [scores]} das granularidades processadas
    @param metrica - Métrica de seleção usada (variancia ou entropia)
    @param caminho_saida - Path para salvar o arquivo de imagem
'''
def plotar_comparativo_granularidade(video_info: dict, dados_escalas: dict, metrica: str, caminho_saida: Path):
    if not dados_escalas:
        return

    # Configurações de estilo para publicação acadêmica
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Cores fixas para garantir consistência visual entre diferentes gráficos
    cores_map = {30: '#440154', 60: '#31688e', 180: '#35b779', 'global': 'red'}
    
    # Escalas de interesse para a RQ01: Detalhe (30s), Padrão (60s) e Diluição (180s)
    escalas_ordenadas = [180, 60, 30, 'global']

    # Converte a duração real de segundos para minutos para o eixo X
    # Baseado no valor retornado por converter_duracao_iso_para_segundos
    duracao_minutos_real = int(video_info.get('duracao', 0)) / 60

    if duracao_minutos_real == 0:
        # Fallback caso a duração não esteja disponível (estimativa por fatias)
        duracao_minutos_real = len(video_info['scores'])

    for g in escalas_ordenadas:
        if g not in dados_escalas: continue
        
        scores = dados_escalas[g]
        
        if g == 'global':
            # Linha tracejada horizontal representando a média total do vídeo
            x_axis = [0, duracao_minutos_real]
            y_axis = [scores[0], scores[0]]
            label = f"Global (Média Total: {scores[0]:.3f})"
            ax.plot(x_axis, y_axis, label=label, color=cores_map[g], linestyle='--', linewidth=3, zorder=5)
        else:
            # Cálculo dos intervalos de tempo baseado na granularidade g
            intervalo_minutos = int(g) / 60
            x_axis = np.arange(len(scores)) * intervalo_minutos
            y_axis = scores
            label = f"Janela: {g}s"
            
            # Ajuste de opacidade e Z-order para evitar que janelas grandes escondam as menores
            alfa = 0.9 if g == 30 else 0.6
            largura = 2.0 if g == 30 else 1.5
            
            # O uso do drawstyle 'steps-post' é crucial para mostrar a persistência da toxicidade no tempo
            ax.plot(x_axis, y_axis, label=label, color=cores_map[g], linewidth=largura, alpha=alfa, zorder=4 if g == 30 else 3)

    # Thresholds fundamentados na análise da natureza dos dados (0.20 e 0.80)
    ax.axhline(0.80, color='black', linestyle=':', alpha=0.4, label='Threshold Tóxico (0.80)')
    ax.axhline(0.20, color='black', linestyle=':', alpha=0.4, label='Threshold Neutro (0.20)')

    # Formatação de Títulos e Legendas
    metrica_label = 'Variância' if metrica == 'variancia' else 'Entropia'

    parte_decimal, parte_inteira = math.modf(duracao_minutos_real)

    plt.title(f"Impacto da Granularidade na Detecção - {metrica_label}\n{video_info['youtuber']} ({video_info['id']} - {int(parte_inteira)}m {int(parte_decimal * 60)}s)", 
              fontsize=14, fontweight='bold')
    
    ax.set_xlabel("Tempo do Vídeo (Minutos)", fontsize=11)
    ax.set_ylabel("Score de Toxicidade (Detoxify)", fontsize=11)
    
    # Define o limite do gráfico exatamente na duração real do vídeo
    ax.set_xlim(0, duracao_minutos_real)
    ax.set_ylim(-0.05, 1.05)
    
    # Posicionamento da legenda fora da área de plotagem para não obstruir os dados
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Escalas Temporais Selecionadas")

    plt.tight_layout()
    
    # Garantir que o diretório de saída exista antes de salvar
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plt.savefig(caminho_saida, bbox_inches='tight')
        plt.close()
    except Exception as e:
        console.print(f"[red]Erro ao salvar gráfico: {e}[/red]")

'''
    Função para orquestrar o fluxo completo de seleção, coleta e plotagem para um grupo específico

    @param grupo_nome - Nome do grupo (Geral, Minecraft, Roblox)
    @param metrica - Métrica para seleção (variancia/entropia)
'''
def orquestrar_plotagem_exemplo(grupo_nome: str, metrica: str):
    console.print(f'\n[green]========== Grupo: {grupo_nome} | Métrica: {metrica} ==========[/green]')

    # Filtragem correta da lista de youtubers baseada no mapa
    if grupo_nome == 'Geral':
        lista_yt = list(MAPA_YOUTUBERS_CATEGORIA.keys())
    else:
        lista_yt = [yt for yt, cat in MAPA_YOUTUBERS_CATEGORIA.items() if cat == grupo_nome]

    # Busca do vídeo que melhor representa a variabilidade do grupo
    video_exemplo = buscar_video_exemplo(lista_yt, metrica)

    if video_exemplo:
        # Carregamento de todas as escalas temporais processadas
        escalas = [30, 60, 120, 180, 240, 300, 'global']
        dados = coletar_dados_granularidade_exemplo(video_exemplo, escalas)
        
        # Resumo estatístico para validação no console
        console.print("\n[bold]Resumo por Escala:[/bold]")
        for esc, scs in dados.items():
            max_val = max(scs)
            status = "[red][TÓXICO][/red]" if max_val >= 0.8 else "[green][NEUTRO][/green]"
            console.print(f"Escala: {str(esc).ljust(6)}s | Máx: {max_val:.4f} | Média: {np.mean(scs):.4f} | {status}")

        # Definição do caminho de saída organizado por grupo
        nome_arquivo = f"SVT_{grupo_nome}_{metrica}_{video_exemplo['id']}.png"
        output_path = BASE_FOLDER / 'SVT' / grupo_nome / nome_arquivo
        
        # Geração do gráfico final
        plotar_comparativo_granularidade(video_exemplo, dados, metrica, output_path)
        console.print(f"\n[green]Sucesso para {grupo_nome}![/green]")
    else:
        console.print(f"[yellow]Nenhum exemplo válido para {grupo_nome}.[/yellow]")

if __name__ == '__main__':
    # Definição dos escopos de análise
    grupos_analise = ['Geral', 'Minecraft', 'Roblox']
    
    # Execução do pipeline de visualização para cada grupo
    for grupo in grupos_analise:
        orquestrar_plotagem_exemplo(grupo, metrica='variancia')
        # orquestrar_plotagem_exemplo(grupo, metrica='entropia')
    
    console.print(f'\n[bold green]Pipeline de análise de granularidade finalizado.[/bold green]')