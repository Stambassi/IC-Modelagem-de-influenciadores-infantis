import pandas as pd
import os
import json
import sys
import numpy as np
import shutil
from rich.console import Console
from rich.rule import Rule

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
    Função global para verificar se existe conteúdo real de forma segura
'''
def tem_conteudo_valido(val):
    if val is None: 
        return False
    
    if isinstance(val, bool): 
        return val
    
    if isinstance(val, (list, dict, set, tuple)): 
        return len(val) > 0
    
    if isinstance(val, np.ndarray): 
        return val.size > 0
    
    try:
        if pd.isna(val): 
            return False
    except: 
        pass

    if isinstance(val, str):
        val_limpo = val.strip().lower()
        if val_limpo in ["", "nan", "none", "null", "[]", "{}"]: 
            return False

    return True

'''
    Remove caracteres especiais de nomes de arquivos para evitar erros no sistema operacional
    @param nome - String original
    @return str - String limpa
''' 
def limpar_nome_arquivo(nome: str) -> str:
    return "".join([c for c in nome if c.isalpha() or c.isdigit() or c in " .-_"]).strip()

'''
    Função para extrair e persistir dados pesados (payload) em arquivos externos,
    evitando que esses dados fiquem apenas em memória ou embutidos no Parquet

    @param video_id - ID único do vídeo
    @param video_data - Dicionário contendo os dados completos do vídeo
    @param base_path - Diretório base para armazenamento (data/{youtuber}/payload)
    @return dict - Caminhos dos arquivos gerados
''' 
def extrair_payload_externo(video_id: str, video_data: dict, base_path: str) -> dict:
    paths = {}
    dirs = {
        "transcript": os.path.join(base_path, "transcripts"),
        "audios": os.path.join(base_path, "audios"),
        "comments": os.path.join(base_path, "comments"),
        "tiras": os.path.join(base_path, "tiras"),
        "analysis": os.path.join(base_path, "analysis"),
    }

    for dir in dirs.values():
        os.makedirs(dir, exist_ok=True)

    # Transcript (JSON)
    transcript = video_data.get("transcript")
    if tem_conteudo_valido(transcript):
        path = os.path.join(dirs["transcript"], f"{video_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, cls=NumpyEncoder)
            paths["transcript_path"] = path.replace("\\", "/") # Padrão universal
        except Exception as e:
            console.print(f"[red]Erro ao salvar transcript {video_id}: {e}[/red]")

    # Áudio (MP3)
    audio_src = video_data.get("audio_local_path")
    if audio_src and os.path.exists(audio_src):
        path = os.path.join(dirs["audios"], f"{video_id}.mp3")
        try:
            shutil.copy2(audio_src, path)
            paths["audio_path"] = path.replace("\\", "/")
        except Exception as e:
            console.print(f"[red]Erro ao salvar áudio {video_id}: {e}[/red]")

    # Comments (CSV)
    comments = video_data.get("comment_data")
    if isinstance(comments, list) and len(comments) > 0:
        path = os.path.join(dirs["comments"], f"{video_id}.csv")
        try:
            pd.DataFrame(comments).to_csv(path, index=False, escapechar='\\')
            paths["comments_path"] = path.replace("\\", "/")
        except Exception as e:
            console.print(f"[red]Erro ao salvar comments {video_id}: {e}[/red]")

    # Tiras (CSV)
    tiras = video_data.get("tiras_data")
    if isinstance(tiras, list) and len(tiras) > 0:
        path = os.path.join(dirs["tiras"], f"{video_id}.csv")
        try:
            pd.DataFrame(tiras).to_csv(path, index=False, escapechar='\\')
            paths["tiras_path"] = path.replace("\\", "/")
        except Exception as e:
            console.print(f"[red]Erro ao salvar tiras {video_id}: {e}[/red]")

    # Analysis (CSV)
    analysis = video_data.get("comment_analysis")
    if isinstance(analysis, list) and len(analysis) > 0:
        path = os.path.join(dirs["analysis"], f"{video_id}.csv")
        try:
            pd.DataFrame(analysis).to_csv(path, index=False, escapechar='\\')
            paths["analysis_path"] = path.replace("\\", "/")
        except Exception as e:
            console.print(f"[red]Erro ao salvar analysis {video_id}: {e}[/red]")

    return paths

'''
    Função para extrair métricas leves a partir dos dados do vídeo,
    evitando a necessidade de parsing pesado em etapas como diff

    @param video_data - Dicionário contendo os dados completos do vídeo
    @return dict - Métricas calculadas (flags e contagens)
''' 
def extrair_metricas_video(video_data: dict) -> dict:
    metrics = {}
    transcript = video_data.get("transcript")
    comments = video_data.get("comment_data")
    tiras = video_data.get("tiras_data")
    analysis = video_data.get("comment_analysis")

    metrics["has_transcript"] = tem_conteudo_valido(transcript)
    metrics["has_comments"] = isinstance(comments, list) and len(comments) > 0
    metrics["has_tiras"] = isinstance(tiras, list) and len(tiras) > 0
    metrics["has_analysis"] = isinstance(analysis, list) and len(analysis) > 0
    metrics["has_audio"] = video_data.get("has_audio", False)
    
    # Sonda interna para Toxicidade nas tiras
    has_tox = has_ptox = False
    if metrics["has_tiras"]:
        for tira in tiras:
            if isinstance(tira, dict):
                if tem_conteudo_valido(tira.get('toxicity')): has_tox = True
                if tem_conteudo_valido(tira.get('p_toxicity')): has_ptox = True
                if has_tox and has_ptox: break

    metrics["has_toxicity"] = has_tox
    metrics["has_perspective"] = has_ptox
    metrics["num_comments"] = len(comments) if isinstance(comments, list) else 0
    metrics["num_tiras"] = len(tiras) if isinstance(tiras, list) else 0

    return metrics

'''
    Função para gerar uma representação leve do vídeo contendo apenas
    metadados e referências para arquivos externos (payload)

    @param video_data - Dicionário com dados completos do vídeo
    @param payload_paths - Caminhos dos arquivos externos gerados
    @return dict - Registro leve para armazenamento em Parquet
''' 
def gerar_indice_video(video_data: dict, payload_paths: dict) -> dict:
    campos_pesados = ["transcript", "comment_data", "tiras_data", "comment_analysis", "audio_local_path"]
    
    indice = {k: v for k, v in video_data.items() if k not in campos_pesados}
    indice.update(payload_paths)
    indice.update(extrair_metricas_video(video_data))
    
    return indice

'''
    Lê estrutura de pastas 'files/{youtuber}' e consolida em um DataFrame de Índice

    @param caminho_youtuber - Caminho raiz do youtuber
    @param nome - Nome do youtuber
    @return pd.DataFrame - DataFrame com todos os dados encontrados
''' 
def ler_dados_locais(caminho_youtuber: str, nome_youtuber: str = None, salvar_payload: bool = False) -> pd.DataFrame:
    todos_indices = []
    
    if not os.path.exists(caminho_youtuber):
        return pd.DataFrame()

    console.print(f"[dim]Varrendo estrutura em {caminho_youtuber}...[/dim]")

    # Carrega dados do canal
    channel_data = {}

    path_channel = os.path.join(caminho_youtuber, "channel_info.csv")
        
    if os.path.exists(path_channel):
        try:
            df_channel = pd.read_csv(path_channel)
            if not df_channel.empty:
                channel_data = {f"{PREFIXO_CANAL}{k}": v for k, v in df_channel.iloc[0].to_dict().items()}
        except Exception as e:
            console.print(f"[bold red]Erro ao ler channel info: {e}[/bold red]")

    # Navega por subpastas
    for root, dirs, files in os.walk(caminho_youtuber):
        if "videos_info.csv" in files:
            try:
                video_data = {}
                path_info = os.path.join(root, "videos_info.csv")
                video_data = pd.read_csv(path_info, dtype=str).iloc[0].to_dict()
                
                if 'video_id' in video_data:
                    video_data['video_id'] = str(video_data['video_id']).strip()

                # Lê tiras_video.info
                path_tiras = os.path.join(root, "tiras_video.csv")
                if os.path.exists(path_tiras):
                    video_data['tiras_data'] = pd.read_csv(path_tiras, engine='python', on_bad_lines='skip').to_dict('records')
                else:
                    video_data['tiras_data'] = []

                # Lê comments_info.csv
                path_comments = os.path.join(root, "comments_info.csv")
                if os.path.exists(path_comments):
                    video_data['comment_data'] = pd.read_csv(path_comments, engine='python', on_bad_lines='skip').to_dict('records')
                else:
                    video_data['comment_data'] = []

                # Lê comments_analysis.csv
                path_analysis = os.path.join(root, "comments_analysis.csv")
                if os.path.exists(path_analysis):
                    video_data['comment_analysis'] = pd.read_csv(path_analysis, engine='python', on_bad_lines='skip').to_dict('records')
                else:
                    video_data['comment_analysis'] = []
                    
                # Lê video_text.json
                path_json = os.path.join(root, "video_text.json")
                if os.path.exists(path_json):
                    with open(path_json, 'r', encoding='utf-8') as f:
                        video_data['transcript'] = json.load(f)
                else:
                    video_data['transcript'] = None

                # Lê [ID].mp3
                path_audio = os.path.join(root, f"{video_data['video_id']}.mp3")
                if os.path.exists(path_audio):
                    video_data['audio_local_path'] = path_audio
                    video_data['has_audio'] = True
                else:
                    video_data['has_audio'] = False

                if channel_data:
                    video_data.update(channel_data)

                payload_paths = {}
                if salvar_payload and nome_youtuber:
                    base_payload_path = os.path.join("data", nome_youtuber, "payload")
                    payload_paths = extrair_payload_externo(video_data.get("video_id"), video_data, base_payload_path)

                todos_indices.append(gerar_indice_video(video_data, payload_paths))

            except Exception as e:
                console.print(f"[bold red]Erro na pasta {root}: {e}[/bold red]")

    df_indices = pd.DataFrame(todos_indices)
    
    # Proteção de Pastas Duplicadas Fisicamente
    if not df_indices.empty and 'video_id' in df_indices.columns:
        df_indices['video_id'] = df_indices['video_id'].astype(str).str.strip()
        colunas_vitais = ['has_transcript', 'has_comments', 'has_tiras', 'has_analysis']
        df_indices['_score'] = df_indices[colunas_vitais].sum(axis=1)
        df_indices = df_indices.sort_values('_score', ascending=True)
        df_indices = df_indices.drop_duplicates(subset=['video_id'], keep='last').drop(columns=['_score'])
        
    return df_indices

'''
    ENCODE: Transforma pastas locais na estrutura Payload + Índice Parquet
'''
def encode(nome_youtuber: str, dir_files="files", dir_data="data"):
    path_files = os.path.join(dir_files, nome_youtuber)
    path_index = os.path.join(dir_data, f"{nome_youtuber}_index.parquet")
    
    console.print(Rule(f"Encode: {nome_youtuber}"))
    
    # Constrói a estrutura e gera o índice da varredura local
    df_indices_local = ler_dados_locais(path_files, nome_youtuber, salvar_payload=True)

    if df_indices_local.empty:
        console.print(f"[yellow]Nenhum dado local encontrado para {nome_youtuber}. Pula.[/yellow]")
        return

    # Garante tipagem segura do ID para o merge
    df_indices_local['video_id'] = df_indices_local['video_id'].astype(str).str.strip()

    # SMART MERGE: Resgatar dados do índice remoto
    if os.path.exists(path_index):
        console.print("[dim]Índice remoto encontrado. Iniciando cruzamento de dados...[/dim]")
        try:
            df_remoto = pd.read_parquet(path_index)
            if not df_remoto.empty and 'video_id' in df_remoto.columns:
                df_remoto['video_id'] = df_remoto['video_id'].astype(str).str.strip()
                
                df_local_idx = df_indices_local.set_index('video_id')
                df_remoto_idx = df_remoto.set_index('video_id')
                
                # 1. Preencher "buracos" dos vídeos que existem no local e no remoto
                ids_comuns = df_local_idx.index.intersection(df_remoto_idx.index)
                
                # Mapeamento de quais colunas copiar se a flag principal estiver faltando localmente
                conjuntos_payload = [
                    ('has_transcript', ['transcript_path']),
                    ('has_audio', ['audio_path']),
                    ('has_comments', ['comments_path', 'num_comments']),
                    ('has_tiras', ['tiras_path', 'num_tiras', 'has_toxicity', 'has_perspective']),
                    ('has_analysis', ['analysis_path'])
                ]
                
                for vid in ids_comuns:
                    for has_flag, colunas in conjuntos_payload:
                        local_tem = df_local_idx.at[vid, has_flag] if has_flag in df_local_idx.columns else False
                        remoto_tem = df_remoto_idx.at[vid, has_flag] if has_flag in df_remoto_idx.columns else False

                        # 1. Local tem o dado e o Remoto não tem (Update para o Remoto)
                        if local_tem and not remoto_tem:
                            df_local_idx.at[vid, has_flag] = True # Mantém True
                        
                        # 2. Se for o caso de tiras, o Local tem a análise (Toxicity/Perspective) e o Remoto não
                        elif has_flag == 'has_tiras' and local_tem and remoto_tem:
                            tox_l = df_local_idx.at[vid, 'has_toxicity']
                            tox_r = df_remoto_idx.at[vid, 'has_toxicity']
                            per_l = df_local_idx.at[vid, 'has_perspective']
                            per_r = df_remoto_idx.at[vid, 'has_perspective']
                            
                            if (tox_l and not tox_r) or (per_l and not per_r):
                                # Força a atualização dos metadados de análise no índice
                                df_local_idx.at[vid, 'has_toxicity'] = tox_l
                                df_local_idx.at[vid, 'has_perspective'] = per_l
                
                # 2. Resgatar vídeos exclusivos do remoto (que não existiam na pasta local)
                ids_exclusivos = df_remoto_idx.index.difference(df_local_idx.index)
                df_exclusivos = df_remoto_idx.loc[ids_exclusivos]
                
                # Unifica o remoto exclusivo com o local enriquecido
                df_final = pd.concat([df_exclusivos, df_local_idx]).reset_index()
                
                console.print(f"[dim]Merge concluído. {len(ids_exclusivos)} vídeos exclusivos do remoto foram preservados.[/dim]")
            else:
                df_final = df_indices_local
        except Exception as e:
            console.print(f"[red]Erro ao ler remoto para merge: {e}. Sobrescrevendo com local.[/red]")
            df_final = df_indices_local
    else:
        df_final = df_indices_local

    # Sincronização de metadados globais (channel_info e atual_date)
    metadados_globais = ["channel_info.csv", "atual_date.csv"]

    for arquivo in metadados_globais:
        path_src = os.path.join(path_files, arquivo)
        if os.path.exists(path_src):
            os.makedirs(os.path.join(dir_data, nome_youtuber), exist_ok=True)
            shutil.copy2(path_src, os.path.join(dir_data, nome_youtuber, arquivo))
            console.print(f"[dim]Metadado '{arquivo}' sincronizado para a pasta 'data'.[/dim]")

    # Salva o Índice Master Definitivo
    try:
        df_final.to_parquet(path_index, index=False)
        console.print(f"[bold green]✓ Índice salvo em {path_index} ({len(df_final)} vídeos mapeados)[/bold green]")
        console.print(f"[bold green]✓ Payloads extraídos/atualizados para data/{nome_youtuber}/payload/[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Erro Fatal no Encoder ao salvar índice: {e}[/bold red]")

'''
    DECODE: Reconstrói pastas locais com base no Índice Parquet e streaming de payloads
'''
def decode(nome_youtuber: str, dir_files="files", dir_data="data"):
    console.print(Rule(f"Decode: {nome_youtuber}"))

    path_index = os.path.join(dir_data, f"{nome_youtuber}_index.parquet")
    path_destino_base = os.path.join(dir_files, nome_youtuber)

    if not os.path.exists(path_index):
        console.print(f"[red]Índice não encontrado: {path_index}[/red]")
        return

    df = pd.read_parquet(path_index)
    df['published_at_dt'] = pd.to_datetime(df['published_at'], errors='coerce') if 'published_at' in df.columns else None

    lista_ids_processados = []
    arquivos_preservados = 0
    arquivos_baixados = 0

    for row in df.itertuples(index=False):
        try:
            video_id = str(getattr(row, "video_id")).strip()
            lista_ids_processados.append(video_id)

            # Estrutura de pastas
            published_at = getattr(row, "published_at_dt")
            ano = str(published_at.year) if pd.notnull(published_at) else "SemData"
            mes = MAPA_MESES.get(published_at.month, "SemData") if pd.notnull(published_at) else "SemData"
            
            titulo_safe = limpar_nome_arquivo(str(getattr(row, "title", "SemTitulo")))
            nome_pasta = f"{titulo_safe} [{video_id}]" if titulo_safe else f"[{video_id}]"

            path_video = os.path.join(path_destino_base, ano, mes, nome_pasta)
            os.makedirs(path_video, exist_ok=True)

            # Reconstrói videos_info.csv (sempre sobrescreve com a versão do Remoto)
            dados_dict = row._asdict()
            campos_excluir = [
                "transcript_path", "comments_path", "tiras_path", "analysis_path", "audio_path",
                "has_transcript", "has_comments", "has_tiras", "has_analysis", "has_audio",
                "has_toxicity", "has_perspective", "num_comments", "num_tiras", "published_at_dt"
            ]
            dados_info = {k: v for k, v in dados_dict.items() if k not in campos_excluir and not k.startswith(PREFIXO_CANAL)}
            pd.DataFrame([dados_info]).to_csv(os.path.join(path_video, "videos_info.csv"), index=False, escapechar='\\')

            # SMART MERGE: Restaurar arquivos pesados apenas se não existirem localmente
            
            # Transcrição            
            target_transcript = os.path.join(path_video, "video_text.json")
            target_audio = os.path.join(path_video, f"{video_id}.mp3")
            
            transcript_path = getattr(row, "transcript_path", None)
            audio_path = getattr(row, "audio_path", None)
            has_transcript = getattr(row, "has_transcript", False)

            # Lógica de Transcrição
            if transcript_path and os.path.exists(transcript_path):
                if not os.path.exists(target_transcript):
                    shutil.copy2(transcript_path, target_transcript)
                    arquivos_baixados += 1
                else:
                    arquivos_preservados += 1

            # Lógica de Áudio
            # Se a transcrição existe no payload e está marcada no índice, deleta o áudio
            if has_transcript and transcript_path and os.path.exists(transcript_path):         
                # 1. Remove da pasta local (files/)
                if os.path.exists(target_audio):
                    os.remove(target_audio)
                    console.print(f"[dim]   └── Áudio local removido (transcrição confirmada para {video_id})[/dim]")
                
                # 2. Remove do Payload (data/)
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                    audios_expurgados += 1
                    console.print(f"[bold red]   └── Payload de áudio deletado (Economia de Nuvem): {video_id}[/bold red]")  
            else:
                # Se ainda não tem transcrição, tenta manter/restaurar o áudio para processamento futuro
                if audio_path and os.path.exists(audio_path):
                    if not os.path.exists(target_audio):
                        shutil.copy2(audio_path, target_audio)
                        arquivos_baixados += 1
                    else:
                        arquivos_preservados += 1

            # Tiras
            tiras_path = getattr(row, "tiras_path", None)
            target_tiras = os.path.join(path_video, "tiras_video.csv")
            if tiras_path and os.path.exists(tiras_path):
                if not os.path.exists(target_tiras):
                    shutil.copy2(tiras_path, target_tiras)
                    arquivos_baixados += 1
                else:
                    arquivos_preservados += 1

            # Comentários
            comments_path = getattr(row, "comments_path", None)
            target_comments = os.path.join(path_video, "comments_info.csv")
            if comments_path and os.path.exists(comments_path):
                if not os.path.exists(target_comments):
                    shutil.copy2(comments_path, target_comments)
                    arquivos_baixados += 1
                else:
                    arquivos_preservados += 1

            # Análises
            analysis_path = getattr(row, "analysis_path", None)
            target_analysis = os.path.join(path_video, "comments_analysis.csv")
            if analysis_path and os.path.exists(analysis_path):
                if not os.path.exists(target_analysis):
                    shutil.copy2(analysis_path, target_analysis)
                    arquivos_baixados += 1
                else:
                    arquivos_preservados += 1

        except Exception as e:
            console.print(f"[red]Erro no vídeo {getattr(row, 'video_id', '???')}: {e}[/red]")

    # Registro de Processados
    with open(os.path.join(path_destino_base, "videoProcessados.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(set(lista_ids_processados)))

    # Reconstrói metadados globais (channel_info e atual_date)
    metadados_globais = ["channel_info.csv", "atual_date.csv"]

    for arquivo in metadados_globais:
        path_src = os.path.join(dir_data, nome_youtuber, arquivo)
        if os.path.exists(path_src):
            shutil.copy2(path_src, os.path.join(path_destino_base, arquivo))
            console.print(f"[dim]Metadado '{arquivo}' restaurado para a pasta local.[/dim]")

    console.print(f"[bold green]Decoder finalizado com sucesso para {nome_youtuber}[/bold green]")
    console.print(f"[dim]Relatório Smart Merge: {arquivos_baixados} novos arquivos baixados | {arquivos_preservados} arquivos locais preservados.[/dim]\n")

'''
    DIFF: Compara o Índice Remoto ultra leve com a estrutura local
'''
def diff(nome_youtuber: str, dir_files="files", dir_data="data"):
    console.print(Rule(f"Diff : {nome_youtuber}"))

    path_files = os.path.join(dir_files, nome_youtuber)
    path_index = os.path.join(dir_data, f"{nome_youtuber}_index.parquet")

    # Ler dados locais (salvar_payload=False garante que não escreve nada no disco)
    df_local = ler_dados_locais(path_files, nome_youtuber, salvar_payload=False)

    if df_local.empty:
        console.print("[yellow]Sem dados locais.[/yellow]")
        return

    if not os.path.exists(path_index):
        console.print("[red]Índice remoto não encontrado. Necessário rodar Encode.[/red]")
        return

    df_remoto = pd.read_parquet(path_index)

    # 1. Conjuntos de IDs
    ids_local = set(df_local['video_id'])
    ids_remoto = set(df_remoto['video_id'])

    novos_locais = ids_local - ids_remoto
    apenas_remoto = ids_remoto - ids_local
    em_ambos = ids_local & ids_remoto

    # Contagens Local
    t_loc = len(df_local)
    trans_loc = df_local['has_transcript'].sum() if 'has_transcript' in df_local.columns else 0
    audio_loc = df_local['has_audio'].sum() if 'has_audio' in df_local.columns else 0
    comm_loc = df_local['has_comments'].sum() if 'has_comments' in df_local.columns else 0
    anal_loc = df_local['has_analysis'].sum() if 'has_analysis' in df_local.columns else 0
    tiras_loc = df_local['has_tiras'].sum() if 'has_tiras' in df_local.columns else 0
    tox_loc = df_local['has_toxicity'].sum() if 'has_toxicity' in df_local.columns else 0
    ptox_loc = df_local['has_perspective'].sum() if 'has_perspective' in df_local.columns else 0

    # Contagens Remoto
    t_rem = len(df_remoto)
    trans_rem = df_remoto['has_transcript'].sum() if 'has_transcript' in df_remoto.columns else 0
    audio_rem = df_remoto['has_audio'].sum() if 'has_audio' in df_remoto.columns else 0
    comm_rem = df_remoto['has_comments'].sum() if 'has_comments' in df_remoto.columns else 0
    anal_rem = df_remoto['has_analysis'].sum() if 'has_analysis' in df_remoto.columns else 0
    tiras_rem = df_remoto['has_tiras'].sum() if 'has_tiras' in df_remoto.columns else 0
    tox_rem = df_remoto['has_toxicity'].sum() if 'has_toxicity' in df_remoto.columns else 0
    ptox_rem = df_remoto['has_perspective'].sum() if 'has_perspective' in df_remoto.columns else 0

    # Exibição (Com Porcentagens Globais)
    console.print(f"[bold]Local: {t_loc} vídeos[/bold]")
    console.print(f"  ├─ Transcrições: [cyan]{trans_loc} ({(trans_loc/t_loc*100) if t_loc>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Áudios: [cyan]{audio_loc} ({(audio_loc/t_loc*100) if t_loc>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Comentários: [cyan]{comm_loc} ({(comm_loc/t_loc*100) if t_loc>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Análises: [cyan]{anal_loc} ({(anal_loc/t_loc*100) if t_loc>0 else 0:.1f}%)[/cyan]")
    console.print(f"  └─ Tiras: [cyan]{tiras_loc} ({(tiras_loc/t_loc*100) if t_loc>0 else 0:.1f}%)[/cyan]")
    console.print(f"     ├─ c/ Detoxify: [cyan]{tox_loc} ({(tox_loc/tiras_loc*100) if tiras_loc>0 else 0:.1f}% das tiras)[/cyan]")
    console.print(f"     └─ c/ Perspective: [cyan]{ptox_loc} ({(ptox_loc/tiras_loc*100) if tiras_loc>0 else 0:.1f}% das tiras)[/cyan]")

    console.print(f"[bold]Remoto: {t_rem} vídeos[/bold]")
    console.print(f"  ├─ Transcrições: [cyan]{trans_rem} ({(trans_rem/t_rem*100) if t_rem>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Áudios: [cyan]{audio_rem} ({(audio_rem/t_rem*100) if t_rem>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Comentários: [cyan]{comm_rem} ({(comm_rem/t_rem*100) if t_rem>0 else 0:.1f}%)[/cyan]")
    console.print(f"  ├─ Análises: [cyan]{anal_rem} ({(anal_rem/t_rem*100) if t_rem>0 else 0:.1f}%)[/cyan]")
    console.print(f"  └─ Tiras: [cyan]{tiras_rem} ({(tiras_rem/t_rem*100) if t_rem>0 else 0:.1f}%)[/cyan]")
    console.print(f"     ├─ c/ Detoxify: [cyan]{tox_rem} ({(tox_rem/tiras_rem*100) if tiras_rem>0 else 0:.1f}% das tiras)[/cyan]")
    console.print(f"     └─ c/ Perspective: [cyan]{ptox_rem} ({(ptox_rem/tiras_rem*100) if tiras_rem>0 else 0:.1f}% das tiras)[/cyan]")
    console.print("-" * 40)

    # 2. Diff de arquivos físicos (vídeos inteiros)
    if novos_locais: 
        console.print(f"[yellow]Vídeos Novos no Local (Falta Encode): {len(novos_locais)}[/yellow]")
        console.print(f"  Ex: {list(novos_locais)[:3]}...")
    else: 
        console.print("[green]Sincronizado: Nenhum novo vídeo local pendente.[/green]")
        
    if apenas_remoto: 
        console.print(f"[red]Faltando no Local (Necessário Decode): {len(apenas_remoto)}[/red]")

    # 3. Diff de Conteúdo Interno
    atualizacao_local_trans, atualizacao_remota_trans = [], []
    atualizacao_local_audio, atualizacao_remota_audio = [], []
    atualizacao_local_comm, atualizacao_remota_comm = [], []
    atualizacao_local_anal, atualizacao_remota_anal = [], []
    atualizacao_local_tiras, atualizacao_remota_tiras = [], []

    if em_ambos and not df_local.empty and not df_remoto.empty:
        df_l_check = df_local.set_index('video_id')
        df_r_check = df_remoto.set_index('video_id')

        for vid in em_ambos:
            # Transcrição
            has_l_trans = df_l_check.at[vid, 'has_transcript'] if 'has_transcript' in df_l_check.columns else False
            has_r_trans = df_r_check.at[vid, 'has_transcript'] if 'has_transcript' in df_r_check.columns else False
            
            if has_l_trans and not has_r_trans: atualizacao_local_trans.append(vid)
            elif has_r_trans and not has_l_trans: atualizacao_remota_trans.append(vid)

            # Áudio
            has_l_audio = df_l_check.at[vid, 'has_audio'] if 'has_audio' in df_l_check.columns else False
            has_r_audio = df_r_check.at[vid, 'has_audio'] if 'has_audio' in df_r_check.columns else False
            
            # Se a transcrição já existe localmente, o áudio local foi deletado de propósito, então não consideramos pendência remota
            if has_l_audio and not has_r_audio: atualizacao_local_audio.append(vid)
            elif has_r_audio and not has_l_audio and not has_l_trans: atualizacao_remota_audio.append(vid)

            # Comentários
            has_l_comm = df_l_check.at[vid, 'has_comments'] if 'has_comments' in df_l_check.columns else False
            has_r_comm = df_r_check.at[vid, 'has_comments'] if 'has_comments' in df_r_check.columns else False

            if has_l_comm and not has_r_comm: atualizacao_local_comm.append(vid)
            elif has_r_comm and not has_l_comm: atualizacao_remota_comm.append(vid)
            
            # Análises
            has_l_anal = df_l_check.at[vid, 'has_analysis'] if 'has_analysis' in df_l_check.columns else False
            has_r_anal = df_r_check.at[vid, 'has_analysis'] if 'has_analysis' in df_r_check.columns else False

            if has_l_anal and not has_r_anal: atualizacao_local_anal.append(vid)
            elif has_r_anal and not has_l_anal: atualizacao_remota_anal.append(vid)

            has_l_tox = df_l_check.at[vid, 'has_toxicity'] if 'has_toxicity' in df_l_check.columns else False
            has_r_tox = df_r_check.at[vid, 'has_toxicity'] if 'has_toxicity' in df_r_check.columns else False
            has_l_per = df_l_check.at[vid, 'has_perspective'] if 'has_perspective' in df_l_check.columns else False
            has_r_per = df_r_check.at[vid, 'has_perspective'] if 'has_perspective' in df_r_check.columns else False
            
            # Se o local tem análise que o remoto não tem (Falta Encode)
            if (has_l_tox and not has_r_tox) or (has_l_per and not has_r_per):
                atualizacao_local_tiras.append(vid)
            # Se o remoto tem análise que o local não tem (Falta Decode)
            elif (has_r_tox and not has_l_tox) or (has_r_per and not has_l_per):
                atualizacao_remota_tiras.append(vid)

    # Exibição dos Alertas Granulares
    if atualizacao_local_trans:
        console.print(f"\n[cyan]video_text.json (Transcrição) Novo no Local (Falta Encode):[/cyan] {len(atualizacao_local_trans)}")
    if atualizacao_remota_trans:
        console.print(f"[magenta]video_text.json (Transcrição) Novo no Remoto (Pode fazer Decode):[/magenta] {len(atualizacao_remota_trans)}")

    if atualizacao_local_audio:
        console.print(f"\n[cyan][ID].mp3 (Áudio) Novo no Local (Falta Encode):[/cyan] {len(atualizacao_local_audio)}")
    if atualizacao_remota_audio:
        console.print(f"[magenta][ID].mp3 (Áudio) Novo no Remoto aguardando transcrição (Pode fazer Decode):[/magenta] {len(atualizacao_remota_audio)}")

    if atualizacao_local_comm:
        console.print(f"\n[cyan]comments_info.csv (Comentários) Novos no Local (Falta Encode):[/cyan] {len(atualizacao_local_comm)}")
    if atualizacao_remota_comm:
        console.print(f"[magenta]comments_info.csv (Comentários) Novos no Remoto (Pode fazer Decode):[/magenta] {len(atualizacao_remota_comm)}")
        
    if atualizacao_local_anal:
        console.print(f"\n[cyan]comments_analysis.csv (Análises) Novas no Local (Falta Encode):[/cyan] {len(atualizacao_local_anal)}")
    if atualizacao_remota_anal:
        console.print(f"[magenta]comments_analysis.csv (Análises) Novas no Remoto (Pode fazer Decode):[/magenta] {len(atualizacao_remota_anal)}")
        
    if atualizacao_local_tiras:
        console.print(f"\n[cyan]tiras_video.csv (Toxicidade) Nova no Local (Falta Encode):[/cyan] {len(atualizacao_local_tiras)}")
    if atualizacao_remota_tiras:
        console.print(f"[magenta]tiras_video.csv (Toxicidade) Nova no Remoto (Pode fazer Decode):[/magenta] {len(atualizacao_remota_trans)}")
        
    console.print("")  

def obter_todos_youtubers(acao: str, dir_files="files", dir_data="data") -> list:
    lista = []
    if acao in ["encode", "diff"] and os.path.exists(dir_files):
        lista = [d for d in os.listdir(dir_files) if os.path.isdir(os.path.join(dir_files, d))]
    
    if acao in ["decode", "diff"] and os.path.exists(dir_data):
        parquets = [f.replace("_index.parquet", "") for f in os.listdir(dir_data) if f.endswith("_index.parquet")]
        lista.extend(parquets)
    
    return sorted(list(set(lista)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        console.print("\n[bold red]Uso: python parquet_converter.py [acao] [nome_youtuber | 'all'][/bold red]")
        sys.exit(1)

    acao = sys.argv[1].lower()
    alvo = sys.argv[2]
    
    lista_youtubers = obter_todos_youtubers(acao) if alvo.lower() in ["all", "todos"] else [alvo]

    if not lista_youtubers:
        console.print("[red]Nenhum youtuber encontrado.[/red]")

    for youtuber in lista_youtubers:
        if acao == "encode": encode(youtuber)
        elif acao == "decode": decode(youtuber)
        elif acao == "diff": diff(youtuber)
        else: console.print(f"[bold red]Ação '{acao}' inválida.[/bold red]"); break