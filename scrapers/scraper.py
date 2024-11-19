import requests
from bs4 import BeautifulSoup

# URL del sito da cui vuoi raccogliere i dati
url = "https://it.wikipedia.org/wiki/Pagina_principale"

# Invia una richiesta HTTP al sito web
response = requests.get(url)

# Verifica se la richiesta è andata a buon fine (codice di stato 200)
if response.status_code == 200:
    # Crea un oggetto BeautifulSoup per analizzare il contenuto HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Estrai tutti i testi contenuti nei paragrafi <p>
    paragraphs = soup.find_all('p')
    for para in paragraphs:
        print(para.get_text())

    with open('output.txt', 'w', encoding='utf-8') as file:
        for para in paragraphs:
            file.write(para.get_text() + '\n')

else:
    print(f"Errore nella richiesta HTTP: {response.status_code}")
