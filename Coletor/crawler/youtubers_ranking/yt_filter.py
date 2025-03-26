import csv

# Link para a referência do Social Blade utilizada: https://socialblade.com/youtube/lists/top/100/subscribers/all/BR?made_for_kids=

# Nome do arquivo de entrada e saída
input_file = "YoutubersGamingTop100.csv" 
output_file = "Minecraft.csv"

# Lista para armazenar os dados extraídos
dados = []

# Leitura e processamento do arquivo
with open(input_file, mode ='r')as file:
  # Ler o arquivo csv
  csvFile = csv.reader(file)

  # Pular a linha do cabeçalho
  next(csvFile)

  # Percorrer cada linha
  for lines in csvFile:
    # Loop para repetir o input até ser válido
    while True:
        # Ler a entrada
        check = input(f'{lines[1]} (S/N): ').strip().upper()

        if check == "S":
            dados.append(lines)
            break  # Sai do loop interno e avança no CSV
        elif check == "N":
            break  # Sai do loop interno e avança no CSV
        else:
            print("Opção inválida! Digite apenas 'S' ou 'N'.")

# Escrevendo no arquivo CSV
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Ranking", "Canal", "Inscritos", "Views", "Vídeos"])
    writer.writerows(dados)

print(f"Arquivo CSV '{output_file}' criado com sucesso!")