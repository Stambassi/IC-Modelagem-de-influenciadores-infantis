from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from rich.console import Console

console = Console()


# Define as regras para cada tipo de análise (métrica)
METRICAS_CONFIG = {
    'sentimento': {
        'coluna_base': 'sentimento_dominante', # Coluna no 'tiras_video.csv' que define o estado
        'tipo_estados': 'categorico', # 'categorico' (ex: POS, NEU) ou 'numerico' (ex: 0.0-1.0)
        'estados': ['POS', 'NEU', 'NEG'] # Lista de estados para análises categóricas
    },
    'negatividade': {
        'coluna_base': 'negatividade',
        'tipo_estados': 'numerico',
        'n_estados': 3 # Número de 'bins' para dividir o score (ex: 3 estados)
    },
    'toxicidade': {
        'coluna_base': 'toxicity',
        'tipo_estados': 'numerico_categorizado', # Novo tipo de estado
        'limiares': [0.0, 0.30, 0.70, 1.01], # 1.01 para garantir que 1.0 seja incluído
        'estados': ['NT', 'GZ', 'T'] # Nomes dos estados
    }
}

'''
    Função para armazenar as transições de estados (para uma métrica específica) de cada vídeo em um arquivo CSV
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3')
'''
def salvar_transicoes_por_metrica(youtubers_list: list[str], metrica_config: dict, nome_analise: str) -> None:
    # Extrair configurações da métrica
    coluna_base = metrica_config['coluna_base']
    tipo_estados = metrica_config['tipo_estados']
    
    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando transições de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan]')

        for tiras_csv_path in base_path.rglob('tiras_video.csv'):            
            video_path = tiras_csv_path.parent
            try:
                (video_path / 'transicoes').mkdir(parents=True, exist_ok=True)
                df_tiras_video = pd.read_csv(tiras_csv_path)

                # Verificar se o DataFrame tem a coluna necessária e dados suficientes
                if df_tiras_video.empty or coluna_base not in df_tiras_video.columns or len(df_tiras_video) < 2:
                    continue

                # Lógica de Definição de Estado
                if tipo_estados == 'categorico':
                    estados = metrica_config['estados']

                    # Renomeia a coluna base (ex: 'sentimento_dominante') para 'estado'
                    df_tiras_video.rename(columns={coluna_base: 'estado'}, inplace=True)

                    # Filtra para garantir que apenas os estados definidos sejam usados
                    df_tiras_video = df_tiras_video[df_tiras_video['estado'].isin(estados)]

                    if len(df_tiras_video) < 2: continue

                    # Converte para tipo Categórico para garantir todas as transições
                    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)
                
                elif tipo_estados == 'numerico':
                    n = metrica_config['n_estados']
                    grupos = np.linspace(0, 1, n + 1)
                    labels = range(1, n + 1)

                    # Usar pd.cut para classificar cada valor da métrica em um estado
                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=grupos,
                        labels=labels,
                        include_lowest=True
                    )

                elif tipo_estados == 'numerico_categorizado':
                    # Usa os limiares e nomes de estados da configuração
                    bins = metrica_config['limiares']
                    labels = metrica_config['estados']

                    # Usar pd.cut para classificar cada valor da métrica em um estado
                    df_tiras_video['estado'] = pd.cut(
                        df_tiras_video[coluna_base],
                        bins=bins,
                        labels=labels,
                        right=False, # Define intervalos como [a, b)
                        include_lowest=True # Garante que 0.0 seja incluído
                    )

                    # Converte para tipo Categórico
                    tipo_categorico = CategoricalDtype(categories=labels, ordered=True)
                    df_tiras_video['estado'] = df_tiras_video['estado'].astype(tipo_categorico)

                    # Re-verificar se há dados suficientes após a filtragem
                    if len(df_tiras_video.dropna(subset=['estado'])) < 2:
                        continue

                else:
                    console.print(f"[red]Erro: Tipo de estado '{tipo_estados}' não reconhecido.[/red]")
                    continue

                # Lógica de transição (comum a ambos os tipos)
                df_tiras_video['proximo_estado'] = df_tiras_video['estado'].shift(-1)
                df_transicoes = df_tiras_video.dropna(subset=['estado', 'proximo_estado'])
                
                # Agrupa e conta, incluindo as transições que não ocorreram (contagem 0)
                contagem = df_transicoes.groupby(['estado', 'proximo_estado'], observed=False).size().reset_index(name='contagem')
                contagem = contagem.sort_values(by=['estado', 'proximo_estado'])

                # Salva no novo arquivo CSV com nome descritivo
                output_path = video_path / 'transicoes' / f'transicoes_{nome_analise}.csv'
                contagem.to_csv(output_path, index=False)
            
            except Exception as e:
                console.print(f'[bold red]Erro[/bold red] em {video_path.name} (salvar_transicoes_por_metrica): {e}')

'''
    Função para criar e persistir a Matriz de Transição (VMG) para cada vídeo individual
    @param youtubers_list - Lista de youtubers a serem analisados
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3')
'''
def salvar_matriz_transicao_video(youtubers_list: list[str], metrica_config: dict, nome_analise: str) -> None:
    # Definir os estados com base no tipo de métrica
    if metrica_config['tipo_estados'] == 'categorico' or metrica_config['tipo_estados'] == 'numerico_categorizado':
        estados = metrica_config['estados'] # Ex: ['POS','NEU','NEG'] ou ['NT','GZ','T']
    else: # numerico
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1))
    
    # Criar o tipo Categórico para garantir a forma da matriz (NxN)
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)

    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matrizes de vídeo de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan]')

        # Procurar recursivamente pelo arquivo de transições de sentimento
        for transicoes_csv_path in base_path.rglob(f'transicoes_{nome_analise}.csv'):                
            try:                
                df_transicoes = pd.read_csv(transicoes_csv_path)
                if df_transicoes.empty:
                    continue

                # Garantir que a matriz final seja sempre NxN
                df_transicoes['estado'] = df_transicoes['estado'].astype(tipo_categorico)
                df_transicoes['proximo_estado'] = df_transicoes['proximo_estado'].astype(tipo_categorico)

                # Calcular a soma das transições que saem de cada estado
                somas_por_estado = df_transicoes.groupby('estado', observed=False)['contagem'].transform('sum')

                # Calcular a probabilidade de cada transição 
                probabilidade = (df_transicoes['contagem'] / somas_por_estado).fillna(0)
                df_transicoes['probabilidade'] = probabilidade

                # Transforma o formato "longo" para o formato "largo" (matriz)
                matriz_transicao = df_transicoes.pivot(
                    index='estado', 
                    columns='proximo_estado', 
                    values='probabilidade'
                )
                
                # Garantir que sejam 0 (caso algum estado nunca seja ponto de partida)
                matriz_transicao.fillna(0, inplace=True)
                
                # Salvar a matriz
                output_folder = transicoes_csv_path.parent.parent / 'VMG'
                output_folder.mkdir(parents=True, exist_ok=True)
                output_path = output_folder / f'VMG_{nome_analise}.csv'
                matriz_transicao.to_csv(output_path)

            except Exception as e:
                console.print(f'Inválido (salvar_matriz_transicao_video): {e}')

'''
    Função para criar e persistir a Matriz de Transição agregada para cada youtuber
    @param youtubers_list - Lista de youtubers a serem analisados.
    @param metrica_config - Dicionário de configuração da métrica (da METRICAS_CONFIG).
    @param nome_analise - O nome da análise (ex: 'sentimento', 'negatividade_3').
    @param agg_metrica - Tipo de cálculo da agregação (ex: 'mean', 'standard', 'variation').
'''
def salvar_matriz_transicao_youtuber(youtubers_list: list[str], metrica_config: dict, nome_analise: str, agg_metrica: str = 'mean') -> None:
    # Definir os estados com base no tipo de métrica
    if metrica_config['tipo_estados'] == 'categorico' or metrica_config['tipo_estados'] == 'numerico_categorizado':
        estados = metrica_config['estados'] # Ex: ['POS','NEU','NEG'] ou ['NT','GZ','T']
    else: # numerico
        n = metrica_config['n_estados']
        estados = list(range(1, n + 1))
    
    # Criar o tipo Categórico para garantir a forma da matriz (NxN)
    tipo_categorico = CategoricalDtype(categories=estados, ordered=True)
    
    # Mapeamento da métrica para a função de agregação do Pandas
    agg_funcs = {
        'mean': 'mean',
        'standard': 'std',
        'variation': lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    }

    if agg_metrica not in agg_funcs:
        console.print(f"[bold red]Erro: Métrica de agregação '{agg_metrica}' é inválida.[/bold red]")
        return
    
    agg_func = agg_funcs[agg_metrica]

    # Percorrer youtubers
    for youtuber in youtubers_list:
        base_path = Path(f'files/{youtuber}')

        if not base_path.is_dir():
            continue

        console.print(f'>>> Processando matriz agregada de "{nome_analise}" para [bold cyan]{youtuber}[/bold cyan] (Agregação: {agg_metrica})')

        try:
            # Encontrar e concatenar todas as transições do youtuber
            lista_dfs_transicoes_por_video = []

            for transicoes_csv_path in base_path.rglob(f'transicoes_{nome_analise}.csv'):
                df_video = pd.read_csv(transicoes_csv_path)

                if not df_video.empty:
                    # Usar o nome da pasta pai (diretório do vídeo) como ID
                    video_id_path = transicoes_csv_path.parent.parent / 'videos_info.csv'

                    # Tentar ler o video_id, mas continuar mesmo se falhar
                    try:
                        df_video['video_id'] = pd.read_csv(video_id_path)['video_id'][0]
                    except Exception:
                        df_video['video_id'] = transicoes_csv_path.parent.parent.name

                    lista_dfs_transicoes_por_video.append(df_video)
            
            if not lista_dfs_transicoes_por_video:
                console.print(f"[yellow]Aviso: Nenhum arquivo de transições de '{nome_analise}' encontrado para {youtuber}. Pulando.[/yellow]")
                continue
            
            df_todas_contagens = pd.concat(lista_dfs_transicoes_por_video, ignore_index=True)

            # Garantir que as colunas 'estado' e 'proximo_estado' sejam categóricas
            df_todas_contagens['estado'] = df_todas_contagens['estado'].astype(tipo_categorico)
            df_todas_contagens['proximo_estado'] = df_todas_contagens['proximo_estado'].astype(tipo_categorico)
            
            # Calcular as probabilidades para CADA vídeo individualmente
            somas_por_estado_video = df_todas_contagens.groupby(['video_id', 'estado'], observed=False)['contagem'].transform('sum')
            df_todas_contagens['probabilidade'] = (df_todas_contagens['contagem'] / somas_por_estado_video).fillna(0)

            # Agregar as probabilidades de todos os vídeos usando a métrica escolhida
            df_agregado = df_todas_contagens.groupby(['estado', 'proximo_estado'], observed=False)['probabilidade'].agg(agg_func).reset_index()
            df_agregado.rename(columns={'probabilidade': agg_metrica}, inplace=True)
            
            # Pivotar a tabela para criar a matriz de transição
            matriz_transicao_youtuber = df_agregado.pivot(
                index='estado', 
                columns='proximo_estado', 
                values=agg_metrica
            )
            
            # Preencher possíveis NaNs com 0.
            matriz_transicao_youtuber.fillna(0, inplace=True)
            
            # Salvar na pasta 'VMG' do youtuber
            output_folder = base_path / 'VMG'
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f'VMG_{nome_analise}_{agg_metrica}.csv'
            
            matriz_transicao_youtuber.to_csv(output_path)
            console.print(f"Matriz de Transição ({agg_metrica}) salva em: {output_path}")

        except Exception as e:
            console.print(f'Inválido (salvar_matriz_transicao_youtuber): {e}')

'''
    Função principal para orquestrar o pipeline de análise de VMG
    @param youtubers_list - Lista de youtubers a serem analisados
    @param config_analise - Dicionário de configurações da métrica (da METRICAS_CONFIG)
    @param nome_analise - O nome identificador desta análise (ex: 'sentimento')
'''
def rodar_pipeline_vmg(youtubers_list: list[str], config_metrica: dict, nome_analise: str):
    console.print(f"\n[bold magenta]===== INICIANDO PIPELINE VMG PARA '{nome_analise.upper()}' =====[/bold magenta]")
    
    # Gerar arquivos de contagem de transições (por vídeo)
    salvar_transicoes_por_metrica(youtubers_list, config_metrica, nome_analise)
    
    # Gerar matrizes de probabilidade (por vídeo)
    salvar_matriz_transicao_video(youtubers_list, config_metrica, nome_analise)
    
    # Gerar matriz agregada (por youtuber) para diferentes agregações
    for agg in ['mean', 'standard', 'variation']:
        salvar_matriz_transicao_youtuber(youtubers_list, config_metrica, nome_analise, agg_metrica=agg)
    
    console.print(f"\n[bold magenta]===== PIPELINE VSMG PARA '{nome_analise.upper()}' CONCLUÍDO =====[/bold magenta]")


if __name__ == '__main__':
    lista_youtubers = ['Robin Hood Gamer', 'Julia MineGirl', 'Tex HS']

    # # Executa o pipeline para a métrica 'sentimento'
    rodar_pipeline_vmg(
        lista_youtubers, 
        config_metrica=METRICAS_CONFIG['sentimento'], 
        nome_analise='sentimento'
    )
    
    # # Executa o pipeline para a métrica 'negatividade' com 3 estados
    # rodar_pipeline_vmg(
    #     lista_youtubers, 
    #     config_metrica=METRICAS_CONFIG['negatividade'], 
    #     nome_analise='negatividade'
    # )
    
    # Executa o pipeline para 'toxicidade' com 3 estados
    rodar_pipeline_vmg(
        lista_youtubers, 
        config_metrica=METRICAS_CONFIG['toxicidade'], 
        nome_analise='toxicidade' 
    )