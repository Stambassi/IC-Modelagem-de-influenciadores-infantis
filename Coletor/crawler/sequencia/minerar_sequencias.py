import pandas as pd
from pathlib import Path
from rich.console import Console
from prefixspan import PrefixSpan

console = Console()

# Configuração global
BASE_DATA_FOLDER = Path('files/sequencias')

'''
    Função para ler o arquivo CSV de sequências precursoras e formatá-lo para o algoritmo

    @param caminho_arquivo - O Path para o arquivo dataset_sequencias_*.csv
    @return list[list[str]] - O banco de dados de sequências (lista de listas)
'''
def carregar_sequencias_precursoras(caminho_arquivo: Path) -> list[list[str]]:
    try:
        # Lê o CSV sem cabeçalho, pois é apenas uma matriz de estados
        df = pd.read_csv(caminho_arquivo, header=None)
        
        # Converte o DataFrame em uma lista de listas
        # .astype(str) garante que tudo seja tratado como texto
        lista_sequencias = df.astype(str).values.tolist()
        
        return lista_sequencias
    except Exception as e:
        console.print(f"     [red]Erro ao carregar sequências de {caminho_arquivo.name}: {e}[/red]")
        return []

'''
    Função que aplica o algoritmo PrefixSpan para encontrar padrões frequentes

    @param database_sequencias - A lista de listas carregada anteriormente
    @param min_suporte_percent - A porcentagem mínima de ocorrência para um padrão ser considerado (0 a 100)
    @return list - Lista de resultados brutos do PrefixSpan [(frequencia, padrao), ...]
'''
def minerar_padroes_frequentes(database_sequencias: list[list[str]], min_suporte_percent: float) -> list:
    total_sequencias = len(database_sequencias)
    
    # Calcula o suporte absoluto (número inteiro de ocorrências)
    min_suporte_absoluto = int(total_sequencias * (min_suporte_percent / 100))
    
    # Proteção para bases muito pequenas
    if min_suporte_absoluto < 1:
        min_suporte_absoluto = 1

    console.print(f"     Minerando com suporte mínimo de: {min_suporte_absoluto} ocorrências ({min_suporte_percent}%)")

    # Inicializar e rodar o PrefixSpan
    ps = PrefixSpan(database_sequencias)
    
    # O método .frequent retorna padrões que atendem ao suporte mínimo. O retorno é uma lista de tuplas: (contagem, [padrao])
    resultados = ps.frequent(min_suporte_absoluto)
    
    return resultados

'''
    Função para formatar os resultados brutos, calcular porcentagens e salvar em um CSV legível

    @param resultados_brutos - A saída do algoritmo PrefixSpan
    @param total_sequencias - O número total de sequências usadas (para calcular %)
    @param caminho_saida - O Path onde o relatório final será salvo
'''
def formatar_e_salvar_resultados(resultados_brutos: list, total_sequencias: int, caminho_saida: Path) -> None:
    if not resultados_brutos:
        console.print(f"     [yellow]Nenhum padrão frequente encontrado com os parâmetros atuais.[/yellow]")
        return

    dados_formatados = []
    
    for contagem, padrao in resultados_brutos:
        # Filtra padrões muito curtos
        if len(padrao) < 1:
            continue

        percentual = (contagem / total_sequencias) * 100
        
        # Transformar a lista ['NT', 'GZ'] em uma string "NT -> GZ" para leitura fácil
        padrao_string = " -> ".join(padrao)
        
        dados_formatados.append({
            'padrao': padrao_string,
            'comprimento': len(padrao),
            'suporte_absoluto': contagem,
            'suporte_percentual': round(percentual, 2)
        })
    
    # Cria DataFrame
    df_resultados = pd.DataFrame(dados_formatados)
    
    # Ordena: Primeiro pelos mais longos, depois pelos mais frequentes (ou vice-versa)
    #df_resultados = df_resultados.sort_values(by=['suporte_percentual', 'comprimento'], ascending=[False, False])
    df_resultados = df_resultados.sort_values(by=['comprimento', 'suporte_percentual'], ascending=[False, False])
    
    # Salva
    df_resultados.to_csv(caminho_saida, index=False)
    console.print(f"     [green]Resultados minerados salvos em:[/green] {caminho_saida}")
    
    # Mostra prévia dos top padrões
    top = df_resultados.head(5)

    i = 1
    for _, row in top.iterrows():
        console.print(f"       Top #{(i)}: [bold]{row['padrao']}[/bold] ({row['suporte_percentual']}%)")
        i += 1

'''
    Função principal para orquestrar a mineração em todos os arquivos preparados

    @param tipo_analise - 'toxicidade' ou 'sentimento' (para filtrar arquivos)
    @param min_suporte - Porcentagem mínima de frequência (padrão 5%)
'''
def orquestrar_mineracao_sequencias(tipo_analise: str, min_suporte: float = 5.0):
    console.print(f"\n[bold magenta]=== Iniciando Mineração de Sequências ({tipo_analise.upper()}) ===[/bold magenta]")
    console.print(f"Buscando gatilhos frequentes (Min Suporte: {min_suporte}%)")

    if not BASE_DATA_FOLDER.is_dir():
        console.print(f"[red]Pasta de sequências não encontrada: {BASE_DATA_FOLDER}[/red]")
        return

    # Busca todos os arquivos dataset_sequencias_*.csv recursivamente
    padrao_busca = f"sequencias_{tipo_analise}_*.csv"
    arquivos_dataset = list(BASE_DATA_FOLDER.rglob(padrao_busca))

    # Verifica se foi encontrado algum dataset
    if not arquivos_dataset:
        console.print(f"[yellow]Nenhum dataset encontrado com o padrão: {padrao_busca}[/yellow]")
        return

    # Itera por todas as pastas de 'sequencias'
    for caminho_dataset in arquivos_dataset:
        # Identificar o grupo/youtuber pelo nome do arquivo ou pasta
        nome_grupo = caminho_dataset.parent.name
        console.print(f"\n  [bold cyan]Minerando padrões para: {nome_grupo}[/bold cyan]")
        
        # Carrega o arquivo das sequências precursoras
        db_sequencias = carregar_sequencias_precursoras(caminho_dataset)
        
        # Veriica se foi encontrado algum arquivo
        if not db_sequencias:
            continue
            
        total_seqs = len(db_sequencias)
        console.print(f"     Base de dados: {total_seqs} sequências antes de um gatilho.")

        # Minera os padrões frequências
        padroes_brutos = minerar_padroes_frequentes(db_sequencias, min_suporte)

        # Salva os resultados
        nome_saida = f"resultados_padroes_{caminho_dataset.name}"
        caminho_saida = caminho_dataset.parent / nome_saida
        
        formatar_e_salvar_resultados(padroes_brutos, total_seqs, caminho_saida)

if __name__ == "__main__":
    MIN_SUPORTE_PERCENT = 10.0 
    
    orquestrar_mineracao_sequencias('toxicidade', MIN_SUPORTE_PERCENT)    
    #orquestrar_mineracao_sequencias('misto_9_estados', MIN_SUPORTE_PERCENT)    
    # orquestrar_mineracao_sequencias('sentimento', MIN_SUPORTE_PERCENT)