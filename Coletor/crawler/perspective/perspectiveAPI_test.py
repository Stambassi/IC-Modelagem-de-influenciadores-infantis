from googleapiclient import discovery
import json

try:
  with open("perspective/api_key.txt", "r") as file:
    API_KEY = file.read()
    
    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=API_KEY,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
    )

    analyze_request = {
      'comment': { 'text': 'friendly greetings from python' },
      'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    print(json.dumps(response, indent=2))
except FileNotFoundError:
  print("Crie um arquivo api_key.txt e coloque a chave")