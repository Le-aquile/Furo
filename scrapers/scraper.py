import requests
from bs4 import BeautifulSoup

def scrape__page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        with open('output.txt', 'w', encoding='utf-8') as file:
            for para in paragraphs:
                file.write(para.get_text() + '\n')
    else:
        print(f"Errore nella richiesta HTTP: {response.status_code}")



