## 📦 Ferramenta de Sincronização de Dados (Parquet Converter)

Este script (parquet_converter.py) é responsável por gerenciar o ciclo de vida dos dados dos YouTubers coletados. Ele atua como uma ponte entre a estrutura de pastas legível por humanos (CSV/JSON locais) e o arquivo otimizado para armazenamento/versionamento (Parquet).

Esta ferramenta garante a integridade dos tipos de dados (evitando que IDs virem números e cortem zeros à esquerda) e resolve problemas de compatibilidade entre bibliotecas (Pandas, Numpy e JSON).

## 🚀 Instalação e Dependências

Para executar este script, você precisa do Python 3.10+ e das bibliotecas listadas abaixo.

### Instalar dependências

Execute o comando no terminal:

```
pip install pandas numpy rich pyarrow fastparquet
```

+ pandas & numpy: Manipulação de dados.
+ rich: Logs coloridos e formatados no terminal.
+ pyarrow: Motor para leitura/escrita de arquivos Parquet.

## 📂 Estrutura de Pastas Esperada

O script assume que o projeto segue a seguinte estrutura:

```
projeto/
│
├── parquet_converter.py   # O script principal
│
├── files/                 # ONDE FICAM AS PASTAS LOCAIS (CSV/JSON)
│   ├── jazzghost/
│   │   ├── channel_info.csv
│   │   ├── 2024/
│   │   │   └── Janeiro/
│   │   │       └── Titulo_Video/
│   │   │           ├── videos_info.csv
│   │   │           ├── tiras_video.csv
│   │   │           └── video_text.json
│   └── authenticgames/
│       └── ...
│
└── data/                  # ONDE FICAM OS ARQUIVOS PARQUET (COMPACTADOS)
    ├── jazzghost.parquet
    └── authenticgames.parquet
```

## 🛠️ Como Usar

A sintaxe básica do comando é:

```
python parquet_converter.py [ACAO] [ALVO]
```

+ **[ACAO]**: encode, decode ou diff.

+ **[ALVO]**: O nome da pasta do youtuber (ex: jazzghost) ou all para processar todos.

## 📖 Cenários de Uso (Passo a Passo)

### Cenário 1: "Acabei de rodar o coletor ou editei arquivos manualmente" (ENCODE)

Você baixou novos vídeos ou corrigiu uma transcrição manualmente na pasta files/ e precisa salvar isso no arquivo Parquet para não perder.

**O que o código faz**: Lê a pasta files/, limpa os tipos de dados e salva em data/{youtuber}.parquet.

```
# Para um youtuber específico
python parquet_converter.py encode "Julia MineGirl"

# Para todos os youtubers (Batch)
python parquet_converter.py encode all
```

**Nota**: Se o arquivo Parquet já existir, ele fará um "Merge", atualizando apenas os vídeos novos ou modificados e mantendo o histórico antigo.

### Cenário 2: "Baixei atualizações do Git/DVC e preciso ver os arquivos" (DECODE)

Você fez um git pull ou dvc pull e recebeu um arquivo .parquet novo, mas precisa ver os JSONs e CSVs na pasta para análise humana.

**O que o código faz**: Lê o arquivo data/{youtuber}.parquet e recria toda a árvore de pastas em files/.

```
# Para um youtuber específico
python parquet_converter.py decode "Julia MineGirl"

# Para todos os youtubers
python parquet_converter.py decode all
```

**Nota**: Isso sobrescreverá os arquivos locais com a versão que está no Parquet.

### Cenário 3: "Quero saber se está tudo sincronizado" (DIFF)

Antes de fazer Encode ou Decode, você quer saber se há diferenças entre o que está na sua pasta e o que está salvo no arquivo.

**O que o código faz**: Compara os IDs dos vídeos na pasta local vs. arquivo Parquet.

```
python parquet_converter.py diff "Julia MineGirl"
```

Saída esperada:

+ 🟢 **Sincronizado**: Tudo igual.
+ 🟡 **Novos Locais**: Você tem vídeos na pasta que ainda não foram salvos no Parquet (Requer Encode).
+ 🔴 **Faltando Local**: O Parquet tem vídeos que você não tem na pasta (Requer Decode).

## ⚙️ Detalhes Técnicos (Para Desenvolvedores)

### Tratamento de Tipos (Sanitização)

Um dos maiores problemas em CSVs é que IDs como "00123" são lidos como o número 123. Este script possui uma função de Sanitização Inteligente:

+ **Herança**: Se já existe um Parquet, ele força os novos dados a seguirem os tipos antigos.
+ **Proteção de IDs**: Colunas terminadas em _id são forçadas como String.
+ **Inferência**: Tenta converter colunas numéricas e booleanas automaticamente.
+ **Fallback**: Se tudo falhar, converte para String para evitar erro fatal (PyArrow Crash).

### NumpyEncoder

O Pandas frequentemente usa tipos do Numpy (int64, float32) que a biblioteca json padrão do Python não consegue salvar. O script utiliza uma classe personalizada NumpyEncoder para converter esses tipos nativamente durante a geração dos arquivos JSON no processo de Decode.

## ⚠️ Solução de Problemas Comuns

### Erro: "ArrowInvalid: ... conversion failed"

Isso significa que uma coluna tem dados misturados (ex: número e texto).

**Solução**: O script tentará corrigir automaticamente usando a função sanitizar_dataframe_generico. Se falhar, ele entrará em Modo de Emergência e salvará tudo como texto para garantir que os dados não sejam perdidos.

### Erro: "Arquivo não encontrado" no Decode

Verifique se o arquivo .parquet existe na pasta data/. Se você usa DVC, lembre-se de rodar dvc pull antes.