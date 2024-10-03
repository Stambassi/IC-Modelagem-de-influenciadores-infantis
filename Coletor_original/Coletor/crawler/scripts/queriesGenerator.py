import re


data = {
  "product": (
    "vape",
    "pod",
    "cigarro eletrônico",
    "vaper",
  ),
  "brand": (
    "voopoo",
    "vaporesso",
    "smok",
    "nikbar"
  ),
  "complement": (
    "faz mal?",
    "review",
    "unboxing",
    "impactos na saúde",
    "vicia?",
    "causa dependência?",
    "ajuda a parar de fumar",
    "tem muita nicotina?",
    "ajuda a largar o cigarro?"
  )
}

# TEMPLATES NÃO PODEM CONTER A MESMA VARIÁVEL MAIS DE UMA VEZ
templates = [
  "[product] [brand] [complement]",
  "[product] [complement]"
]

def extract_variables(string):
  regex = r"\[(\w+)\]"
  variables = re.findall(regex, string)
  return variables

def generate_single_template(template, data):
  temp = [template]
  temp_2 = []

  n = len(extract_variables(template))

  for element in data:
    for string in temp:
      for value in data[element]:
        temp_2.append(string.replace(f"[{element}]", value))
    
    for item in temp_2:
      temp.append(item)
    for item in temp:
      if(len(extract_variables(item)) == n):
        temp.remove(item)
    n -= 1
    temp_2 = []

  res = []

  for item in temp:
    if len(extract_variables(item)) == 0 : 
      res.append(item)
  
  return res

# Pode sobrecarregar o limite de pesquisas da API
def generate_queries():
  # Inserção manual de pesquisas podem ser realizadas dentro da lista queries
  queries = [
        '"Cigarro eletrônico" faz mal?"', '"vape" faz mal?',
        '"Cigarro eletrônico vicia?"', "vape vicia?",
        "Cigarro eletrônico ajuda a parar de fumar", "vape ajuda a parar de fumar",
        "Cigarro eletrônico tem mais nicotina?", "vape tem mais nicotina?",
        "Cigarro eletrônico ou cigarro normal?", "vape ou cigarro normal",
        "Efeitos do cigarro eletrônico na saúde", "Efeitos do vape na saúde",
        "Riscos do uso de cigarro eletrônico para a saúde", "Riscos do uso de vape para a saúde",
        "Cigarro eletrônico causa doenças?", "vape causa doenças?",
        "Impactos do cigarro eletrônico no pulmão", "Impactos do vape no pulmão",
        "Dependência de cigarro eletrônico", "Dependência de vape",
        "Cigarro eletrônico causa dependência?", "vape causa dependência?",
        "Diferenças na dependência de cigarro eletrônico e cigarro tradicional", "Diferenças na dependência de vape e cigarro tradicional",
        "Cigarro eletrônico é eficaz para largar o cigarro?", "vape é eficaz para largar o cigarro?",
        "Experiências de parar de fumar com cigarro eletrônico", "Experiências de parar de fumar com vape",
        "Composição química do cigarro eletrônico", "Composição química do vape",
        "Concentração de nicotina em cigarros eletrônicos", "Concentração de nicotina em vapes",
        "Diferença entre nicotina no cigarro eletrônico e no cigarro tradicional", "Diferença entre nicotina no vape e no cigarro tradicional"
       # Olhar marcas, termos, perguntas (ex. pod mata, vaper causa cancer?)
       # Olhar as top marcas vendidas no mundo, brasil, procurar notícias, etc....
    ]


  # Para cada template
  for template in templates:
    
    # Cria o dicionario com os dados necessários para o template
    templateData = {} 
    for variable in extract_variables(template):
      templateData[variable] = data[variable]

    # Gera o conjunto de strings para aquele template
    list = generate_single_template(template, templateData)
    queries += list

  return queries