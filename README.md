# Modelagem de influenciadores infantis


Este projeto desenvolvido na PUC-Minas tem como objetivo geral identificar a toxicidade presente em discursos de influenciadores infantis de Minecraft e Roblox. Como objetivos específicos, buscamos: (1) analisar valores atípicos relacionados à toxicidade a fim de detectar picos de discurso negativo, e (2) propor uma metodologia abstrata capaz de identificar oscilações anormais no discurso ao longo do tempo.

A metodologia em desenvolvimento consiste na seleção de influenciadores populares junto ao público infantil, seguida da coleta de vídeos por meio da API do YouTube. Os vídeos serão transcritos com o auxílio de modelos automáticos de transcrição, e suas legendas analisadas por modelos de inteligência artificial capazes de avaliar o grau de toxicidade do conteúdo. Uma abordagem analítica chamada de “algoritmo de tirinhas” será aplicada para segmentar os discursos e identificar padrões críticos de linguagem.

Embora o projeto ainda esteja em andamento, espera-se como resultado a identificação de padrões de comportamento ofensivo em vídeos voltados ao público infantil, que podem comprometer a segurança digital de crianças e adolescentes.

Como conclusão preliminar, destacamos a relevância social e científica da proposta, que poderá subsidiar ferramentas de controle parental, contribuir para a regulação de conteúdo online e, sobretudo, oferecer maior segurança no consumo digital infantil. 


# Tutorial

+ Criar o virtual enviroment: `python3 -m venv venv`
+ Ativá-lo: `source venv/bin/activate`
+ Baixar as bibliotecas necessárias: `pip install -r Coletor/crawler/requirements.txt`
+ Rodar o código principal: `python3 Coletor/crawler/main.py`
+ Ao rodar o main.py, abrirá um menu interativo com as opções do coletor

# Estrutura

```
IC-Modelagem-de-Influenciadores-Infantis
│   README.md
|   Coletor_original.zip    
└── Coletor
    |  
    └── crawler
        │   main.py
        │   config.py
        |   reset.py
        |   script.py
        |   video_process.py
        |   youtuberslist.csv
        |   ...
        |  
        └─ files
           |
           └─ Nome do Canal
                    |
                    └─ 2019 ... 2025
                       └─ Janeiro ... Dezembro
                            └─ Nome do Video
                                | comments_info.csv
                                | comments_analysis.csv
                                | videos_info.csv
                                | video_text.json
```


# Participantes

+ Augusto Stambassi Duarte - Pesquisador [Git Pessoal](https://github.com/stambassi)
+ João Pedro Torres - Pesquisador [Git Pessoal](https://github.com/stambassi)
+ Lucas Carneiro Nassau Malta - Pesquisador [Git Pessoal](https://github.com/stambassi)
+ Humberto Torres Marques Neto - Orientador 


