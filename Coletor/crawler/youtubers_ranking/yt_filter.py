import csv
import math
# Link para a referência do Social Blade utilizada: https://socialblade.com/youtube/lists/top/100/subscribers/all/BR?made_for_kids=

# Nome do arquivo de entrada e saída
input_file = "Gaming.csv"
output_file = "Roblox.csv"
minecraf_file = "Minecraft.csv"
min_inscritos = 5.0
# Lista para armazenar os dados extraídos
dados = []

# Leitura e processamento do arquivo
with open(input_file, mode='r')as file:
    
    arquivo_minecraft = open(minecraf_file, 'r')

    # Ler o arquivo csv
    csvFile = csv.reader(file)
    csvMine = csv.reader(arquivo_minecraft)
    # Pular a linha do cabeçalho
    next(csvFile)
    x = 0

    # Percorrer cada linha
    for lines in csvFile:
        # Loop para repetir o input até ser válido
        inscritos = lines[2][0:len(lines[2])-1]
        if(float(inscritos) > min_inscritos or math.isclose(float(inscritos), min_inscritos, rel_tol=1e-6)):
            while True:
                #print (inscritos)

                    # Ler a entrada
                    check = input(f'{x}: {lines[1]} (S/N): ').strip().upper()

                    if check == "S":
                        dados.append(lines)
                        break  # Sai do loop interno e avança no CSV
                    elif check == "N":
                        break  # Sai do loop interno e avança no CSV
                    else:
                        print("Opção inválida! Digite apenas 'S' ou 'N'.")
            x+=1
        else: 
            print(f"{lines[1]} não tem quantidade minima de inscritos")

    arquivo_minecraft.close()

# Escrevendo no arquivo CSV
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Ranking", "Canal", "Inscritos", "Views", "Vídeos"])
    writer.writerows(dados)

print(f"Arquivo CSV '{output_file}' criado com sucesso!")
