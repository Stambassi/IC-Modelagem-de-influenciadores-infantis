import csv

# Link para a referência do Social Blade utilizada: https://socialblade.com/youtube/lists/top/100/subscribers/all/BR?made_for_kids=

# Nome do arquivo de entrada e saída
input_file = "YoutubersGamingTop100.txt" 
output_file = "YoutubersGamingTop100.csv"

# Lista para armazenar os dados extraídos
dados = []

# Leitura e processamento do arquivo
with open(input_file, "r", encoding="utf-8") as file:
    # Ler todas as linhas do arquivo como um arranjo
    lines = file.readlines()

    # Inicializar variável que conta a posição no arranjo de linhas
    actualLine = 0

    for i in range(100):
        ranking = lines[actualLine][:-3].strip()
        actualLine += 4

        canal = lines[actualLine].strip()
        actualLine += 2

        inscritos = lines[actualLine].strip()
        actualLine += 2

        views = lines[actualLine].strip()
        actualLine += 2

        videosCount = lines[actualLine].strip()
        actualLine += 1

        dados.append([ranking, canal, inscritos, views, videosCount])

# Escrevendo no arquivo CSV
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Ranking", "Canal", "Inscritos", "Views", "Vídeos"])
    writer.writerows(dados)

print(f"Arquivo CSV '{output_file}' criado com sucesso!")