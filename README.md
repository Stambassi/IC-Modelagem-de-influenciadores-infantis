# Modelagem de influenciadores infantis
Coletor de dados do Youtube
Para utilizar o código, deve-se baixar todas as bibliotecas adequadas assim como os modelos de IA utilzados (VOSK, BERTopic) e coloca-los na pasta com seu nome especificado.
Após a preparação, execute o código main.py e selecione "Adicionar novo(s) influenciadores(s)" e depois "Começar a coleta de dados" para executar a coleta.
Para visualização da coleta, todo o dado gerado é inserido dentro na pasta files com a seguinte subdivisão:
```
IC-Modelagem-de-Influenciadores-Infantis
│   README.md    
└── Coletor
    |  
    └── crawler
        │   main.py
        │   config.py
        |   reset.py
        |   script.py
        | 
        └─ files
           |
           └─ Nome do Canal
                    |
                    └─ 2019 ... 2025
                       └─ Janeiro ... Dezembro
                            └─ Nome do Video
                                | channels_info.csv
                                | comments_info.csv
                                | processed_videos.csv
                                | videos_info.csv
```
Projeto de Iniciacao cientifica desenvolvido na PUC Minas
## Participantes
- Humberto Torres Marques Neto
- Augusto Stambassi Duarte
- João Pedro Torres
- Pedro Henrique Felix dos Santos

