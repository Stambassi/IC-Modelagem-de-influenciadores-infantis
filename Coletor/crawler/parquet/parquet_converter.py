import pandas as pd
import os
import json
import sys
import numpy as np
from rich.console import Console
from rich.rule import Rule

# Inicializa o console do Rich para logs bonitos
console = Console()

# Mapeamento para criar pastas com nomes de meses em Português
MAPA_MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# Prefixo usado para separar metadados do canal dos metadados do vídeo no DataFrame
PREFIXO_CANAL = "ch_meta_"

'''
    Classe auxiliar para ensinar o json.dump a lidar com tipos do NumPy
    Motivo: O Pandas/Numpy usam tipos próprios (int64, float32) que o JSON nativo do Python não aceita
'''
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)

'''
    Remove caracteres especiais de nomes de arquivos para evitar erros no sistema operacional
    @param nome - String original
    @return str - String limpa
'''
def limpar_nome_arquivo(nome: str) -> str:
    return "".join([c for c in nome if c.isalpha() or c.isdigit() or c in " .-_"]).strip()

'''
    Função vital para compatibilidade com Parquet (PyArrow)
    O Pandas é flexível com tipos mistos, mas o Parquet é estrito.
    Esta função tenta converter colunas para números/booleanos e, se falhar, garante texto
    
    @param df_novo - DataFrame com dados locais
    @param df_referencia - (Opcional) DataFrame remoto para copiar o schema
    @return pd.DataFrame - DataFrame higienizado
'''
def sanitizar_dataframe_generico(df_novo: pd.DataFrame, df_referencia: pd.DataFrame = None) -> pd.DataFrame:
    df_final = df_novo.copy()
    
    # Herança de Tipos: Se já tem dados antigos, tenta imitar os tipos deles
    if df_referencia is not None:
        for col in df_referencia.columns:
            if col in df_final.columns:
                try:
                    dtype_ref = df_referencia[col].dtype
                    # Se era numérico, tenta converter o novo para numérico
                    if pd.api.types.is_numeric_dtype(dtype_ref):
                        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                    # Se não for genérico (object), força o tipo específico
                    if dtype_ref != 'object':
                        df_final[col] = df_final[col].astype(dtype_ref)
                except:
                    pass # Se falhar, cairá na regra genérica abaixo

    # Inferência Genérica: Para colunas novas ou sem referência
    for col in df_final.columns:
        # Se já tratamos acima, pula
        if df_referencia is not None and col in df_referencia.columns:
            continue
            
        # Colunas de ID (video_id, channel_id) NUNCA devem ser números
        # para evitar perder zeros à esquerda (ex: "001" virar 1)
        if not col.lower().endswith("id") and not col.lower().endswith("ids"):
            # Tenta converter para número
            df_temp = pd.to_numeric(df_final[col], errors='coerce')
            # Se a conversão funcionou (não gerou apenas Nulos), é aceita
            if not df_temp.isna().all():
                df_final[col] = df_temp
                continue

        # Tenta converter para Booleano (True/False)
        # Verifica se os valores únicos parecem booleanos
        unique_vals = set(df_final[col].astype(str).str.lower().unique())
        is_bool = unique_vals.issubset({'true', 'false', '1', '0', '1.0', '0.0', 'nan', 'none'})
        
        if is_bool:
            mapper = {'true': True, '1': True, '1.0': True, 'false': False, '0': False, '0.0': False}
            df_final[col] = df_final[col].astype(str).str.lower().map(mapper).fillna(False).astype(bool)
            continue

        # Fallback: Se não é número nem bool, garantimos que seja STRING
        df_final[col] = df_final[col].astype(str)
    
    return df_final

'''
    Lê recursivamente a estrutura de pastas 'files/{youtuber}' e consolida em um DataFrame
    
    @param caminho_youtuber - Caminho raiz do youtuber
    @return pd.DataFrame - DataFrame com todos os dados encontrados
'''
def ler_dados_locais(caminho_youtuber: str) -> pd.DataFrame:
    todos_videos = []
    
    if not os.path.exists(caminho_youtuber):
        return pd.DataFrame()

    console.print(f"[bold blue]Lendo estrutura de pastas em:[/bold blue] [cyan]{caminho_youtuber}[/cyan]...")

    # Carrega dados do canal (metadados globais)
    channel_data = {}
    path_channel = os.path.join(caminho_youtuber, "channel_info.csv")
    
    if os.path.exists(path_channel):
        try:
            df_channel = pd.read_csv(path_channel)
            if not df_channel.empty:
                row_channel = df_channel.iloc[0].to_dict()
                channel_data = {f"{PREFIXO_CANAL}{k}": v for k, v in row_channel.items()}
        except Exception as e:
            console.print(f"[bold red]Erro ao ler channel_info.csv: {e}[/bold red]")

    # Navega por todas as subpastas
    for root, dirs, files in os.walk(caminho_youtuber):
        if "videos_info.csv" in files:
            try:
                video_data = {}

                # Ler Metadata
                # dtype=str previne que o Pandas converta IDs numéricos automaticamente
                path_info = os.path.join(root, "videos_info.csv")
                df_info = pd.read_csv(path_info, dtype=str)
                video_data = df_info.iloc[0].to_dict()

                if 'video_id' in video_data:
                    video_data['video_id'] = str(video_data['video_id']).strip()

                # Ler Tiras
                path_tiras = os.path.join(root, "tiras_video.csv")
                if os.path.exists(path_tiras):
                    df_tiras = pd.read_csv(path_tiras)
                    video_data['tiras_data'] = df_tiras.to_dict(orient='records')
                else:
                    video_data['tiras_data'] = []

                # Ler Transcrição
                path_json = os.path.join(root, "video_text.json")
                if os.path.exists(path_json):
                    with open(path_json, 'r', encoding='utf-8') as f:
                        transcript = json.load(f)
                    video_data['transcript'] = transcript
                else:
                    video_data['transcript'] = None

                # Mescla dados do vídeo com dados do canal
                if channel_data:
                    video_data.update(channel_data)

                todos_videos.append(video_data)
            except Exception as e:
                console.print(f"[bold red]Erro ao processar pasta {root}: {e}[/bold red]")

    df = pd.DataFrame(todos_videos)
    # Garante tipagem de ID no final
    if not df.empty and 'video_id' in df.columns:
        df['video_id'] = df['video_id'].astype(str).str.strip()
        
    return df

'''
    ENCODE: Lê pastas locais, mescla com parquet remoto (se houver) e salva tudo
    
    @param nome_youtuber - Nome da pasta
    @param dir_files - Diretório de origem (local)
    @param dir_data - Diretório de destino (parquet)
'''
def atualizar_parquet_com_locais(nome_youtuber: str, dir_files="files", dir_data="data"):
    path_files = os.path.join(dir_files, nome_youtuber)
    path_parquet = os.path.join(dir_data, f"{nome_youtuber}.parquet")
    
    console.print(Rule(f"Encoder: {nome_youtuber}"))
    
    # Carrega o que há localmente
    df_local = ler_dados_locais(path_files)
    
    if df_local.empty:
        console.print(f"[yellow]Nenhum dado local encontrado para {nome_youtuber}. Pula.[/yellow]")
        return
    
    # Carrega o histórico remoto (Parquet existente)
    df_remoto = pd.DataFrame()
    if os.path.exists(path_parquet):
        console.print(f"[blue]Lendo remoto em[/blue] [cyan]{path_parquet}[/cyan]...")
        try:
            df_remoto = pd.read_parquet(path_parquet)
            if 'video_id' in df_remoto.columns:
                df_remoto['video_id'] = df_remoto['video_id'].astype(str).str.strip()
        except Exception as e:
            console.print(f"[bold red]Erro ao ler parquet: {e}[/bold red]")

    # Processo de Mesclagem (Merge)
    if not df_local.empty:
        # Garante ID string
        if 'video_id' in df_local.columns:
             df_local['video_id'] = df_local['video_id'].astype(str).str.strip()

        console.print("[dim]Normalizando tipos de dados...[/dim]")
        
        # Aplica sanitização: Usa o remoto como gabarito de tipos para o local
        if not df_remoto.empty:
            df_local = sanitizar_dataframe_generico(df_local, df_referencia=df_remoto)
        else:
            df_local = sanitizar_dataframe_generico(df_local)

        # Junta tudo e remove duplicatas (mantendo a versão local mais recente)
        df_unificado = pd.concat([df_remoto, df_local])
        df_unificado = df_unificado.drop_duplicates(subset=['video_id'], keep='last')
    else:
        df_unificado = df_remoto

    # Salvamento
    if not df_unificado.empty:
        os.makedirs(dir_data, exist_ok=True)
        try:
            # Última garantia de ID
            if 'video_id' in df_unificado.columns:
                df_unificado['video_id'] = df_unificado['video_id'].astype(str)
                
            df_unificado.to_parquet(path_parquet, index=False)
            console.print(f"[bold green]Sucesso! {path_parquet} salvo com {len(df_unificado)} registros.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Erro Fatal no Encoder: {e}[/bold red]")
            # Fallback de Emergência: Salvar tudo como texto se a tipagem estrita falhar
            console.print("[yellow]Tentando salvar convertendo tudo para String (Modo de Emergência)...[/yellow]")
            df_unificado = df_unificado.astype(str)
            df_unificado.to_parquet(path_parquet, index=False)
            console.print("[green]Salvo em modo texto.[/green]")
    else:
        console.print("[red]Nada para salvar.[/red]")
    console.print("")

'''
    Ação DECODE: Lê arquivo Parquet e recria a árvore de diretórios e arquivos JSON/CSV.
    
    @param nome_youtuber - Nome da pasta
    @param dir_files - Diretório de destino (local)
    @param dir_data - Diretório de origem (parquet)
'''
def recriar_pastas_do_parquet(nome_youtuber: str, dir_files="files", dir_data="data"):
    path_parquet = os.path.join(dir_data, f"{nome_youtuber}.parquet")
    path_destino_base = os.path.join(dir_files, nome_youtuber)

    console.print(Rule(f"Decoder: {nome_youtuber}"))

    if not os.path.exists(path_parquet):
        console.print(f"[bold red]Erro: Arquivo {path_parquet} não encontrado.[/bold red]")
        return

    try:
        df = pd.read_parquet(path_parquet)
    except Exception as e:
        console.print(f"[bold red]Erro ao ler Parquet: {e}[/bold red]")
        return

    if 'video_id' in df.columns:
        df['video_id'] = df['video_id'].astype(str).str.strip()
    
    # Cria coluna auxiliar de data para pastas
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce')

    # Restaura channel_info.csv
    cols_canal = [c for c in df.columns if c.startswith(PREFIXO_CANAL)]
    if cols_canal:
        os.makedirs(path_destino_base, exist_ok=True)
        df_channel_info = df.iloc[[0]][cols_canal].copy()
        df_channel_info.columns = [c.replace(PREFIXO_CANAL, "") for c in df_channel_info.columns]
        path_channel_csv = os.path.join(path_destino_base, "channel_info.csv")
        df_channel_info.to_csv(path_channel_csv, index=False)

    lista_ids_processados = []

    # Processa linha por linha (vídeo por vídeo)
    for _, row in df.iterrows():
        try:
            video_id = str(row['video_id']).strip()
            lista_ids_processados.append(video_id)
            
            # Define estrutura Ano/Mes
            if pd.notnull(row['published_at_dt']):
                ano = str(row['published_at_dt'].year)
                mes_num = row['published_at_dt'].month
                mes = MAPA_MESES.get(mes_num, "Desconhecido")
            else:
                ano = "SemData"
                mes = "SemData"

            titulo_safe = limpar_nome_arquivo(str(row['title']))
            path_video = os.path.join(path_destino_base, ano, mes, titulo_safe)
            os.makedirs(path_video, exist_ok=True)

            # Recriar videos_info.csv (Metadados simples)
            cols_drop_aux = ['tiras_data', 'transcript', 'published_at_dt']
            cols_drop_total = cols_drop_aux + cols_canal
            cols_drop_existentes = [c for c in cols_drop_total if c in df.columns]
            
            dados_info = row.drop(labels=cols_drop_existentes).to_frame().T
            dados_info.to_csv(os.path.join(path_video, "videos_info.csv"), index=False)

            # Recriar tiras_video.csv (Frames)
            tiras_data = row.get('tiras_data')
            if isinstance(tiras_data, str):
                try: tiras_data = json.loads(tiras_data)
                except: tiras_data = []
            
            if tiras_data is not None:
                if isinstance(tiras_data, np.ndarray): tiras_data = tiras_data.tolist()
                if isinstance(tiras_data, list) and len(tiras_data) > 0:
                    df_tiras = pd.DataFrame(tiras_data)
                    df_tiras.to_csv(os.path.join(path_video, "tiras_video.csv"), index=False)

            # Recriar Transcrição (JSON)
            transcript = row.get('transcript')
            
            # Tratamento: Pode vir como string JSON ou Objeto Python
            if isinstance(transcript, str):
                try: transcript_obj = json.loads(transcript)
                except: transcript_obj = None
            else:
                transcript_obj = transcript

            if transcript_obj is not None:
                # Verifica se não é NaN float
                if not (isinstance(transcript_obj, float) and np.isnan(transcript_obj)):
                    # Usa NumpyEncoder para garantir que tipos int64/float32 sejam salvos corretamente
                    with open(os.path.join(path_video, "video_text.json"), 'w', encoding='utf-8') as f:
                        json.dump(transcript_obj, f, ensure_ascii=False, cls=NumpyEncoder)
                        
        except Exception as e:
            console.print(f"[red]Erro ao processar vídeo {row.get('video_id')}: {e}[/red]")

    path_processados = os.path.join(path_destino_base, "videoProcessados.txt")
    with open(path_processados, 'w', encoding='utf-8') as f:
        for vid in set(lista_ids_processados): f.write(f"{vid}\n")

    console.print(f"[bold green]Decoder finalizado para {nome_youtuber}.[/bold green]\n")

'''
    Ação DIFF: Compara IDs e status entre Pasta Local e Parquet Remoto.
'''
def calcular_diferenca(nome_youtuber: str, dir_files="files", dir_data="data"):
    path_parquet = os.path.join(dir_data, f"{nome_youtuber}.parquet")
    path_files = os.path.join(dir_files, nome_youtuber)
    
    console.print(Rule(f"Diff: {nome_youtuber}"))

    # Local
    df_local = ler_dados_locais(path_files)
    ids_local = set()
    if not df_local.empty and 'video_id' in df_local.columns:
        # Remove duplicatas locais para contagem precisa
        df_local = df_local.drop_duplicates(subset=['video_id'], keep='last')
        ids_local = set(df_local['video_id']) 
    
    # Remoto
    ids_remoto = set()
    df_parquet = pd.DataFrame()
    if os.path.exists(path_parquet):
        try:
            df_parquet = pd.read_parquet(path_parquet)
            if not df_parquet.empty and 'video_id' in df_parquet.columns:
                df_parquet['video_id'] = df_parquet['video_id'].astype(str).str.strip()
                ids_remoto = set(df_parquet['video_id'])
        except: pass

    # Operações de Conjunto
    novos_locais = ids_local - ids_remoto
    apenas_remoto = ids_remoto - ids_local

    console.print(f"Vídeos Locais: {len(ids_local)} | Vídeos Remotos: {len(ids_remoto)}")
    
    if novos_locais:
        console.print(f"[yellow]Novos Locais (Falta Encode):[/yellow] {len(novos_locais)}")
    else:
        console.print("[green]Sincronizado: Nenhum novo local pendente.[/green]")
    
    if apenas_remoto:
        console.print(f"[red]Faltando Local (Necessário Decode):[/red] {len(apenas_remoto)}")
    
    console.print("")

'''
    Descobre lista de youtubers baseado nos diretórios ou arquivos existentes
'''
def obter_todos_youtubers(acao: str, dir_files="files", dir_data="data") -> list:
    lista = []
    if acao in ["encode", "diff"]:
        # Olha para as pastas em files/
        if os.path.exists(dir_files):
            lista = [d for d in os.listdir(dir_files) if os.path.isdir(os.path.join(dir_files, d))]
    
    if acao in ["decode", "diff"]:
        # Se for decode, olha os parquets. Se for diff, junta os dois.
        if os.path.exists(dir_data):
            parquets = [f.replace(".parquet", "") for f in os.listdir(dir_data) if f.endswith(".parquet")]
            lista.extend(parquets)
    
    return sorted(list(set(lista)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        console.print("\n[bold red]Uso: python parquet_converter.py [acao] [nome_youtuber | 'all'][/bold red]")
        console.print("  [cyan]encode[/cyan] [all|nome] -> Transforma Pastas Locais em Arquivo Parquet")
        console.print("  [cyan]decode[/cyan] [all|nome] -> Explode Arquivo Parquet em Pastas Locais")
        console.print("  [cyan]diff[/cyan]   [all|nome] -> Relatório de Diferença entre Local e Remoto")
        sys.exit(1)

    acao = sys.argv[1].lower()
    alvo = sys.argv[2]
    
    lista_youtubers = []

    # Modo Batch: Se usuário digitar "all", processa todos encontrados
    if alvo.lower() in ["all", "todos"]:
        console.print(f"[bold yellow]Modo Batch ativado: Processando TODOS para ação '{acao}'...[/bold yellow]")
        lista_youtubers = obter_todos_youtubers(acao)
        if not lista_youtubers:
            console.print("[red]Nenhum youtuber encontrado nos diretórios.[/red]")
    else:
        # Modo Individual
        lista_youtubers = [alvo]

    # Execução
    for youtuber in lista_youtubers:
        if acao == "encode":
            atualizar_parquet_com_locais(youtuber)
        elif acao == "decode":
            recriar_pastas_do_parquet(youtuber)
        elif acao == "diff":
            calcular_diferenca(youtuber)
        else:
            console.print(f"[bold red]Ação '{acao}' inválida.[/bold red]")
            break