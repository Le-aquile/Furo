import requests
from bs4 import BeautifulSoup

def scrape__page(url, save_to_file=False):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        if save_to_file:
            with open('output.txt', 'w', encoding='utf-8') as file:
                for para in paragraphs:
                    file.write(para.get_text() + '\n')
        else:
            return [para.get_text() for para in paragraphs]
    else:
        print(f"Errore nella richiesta HTTP: {response.status_code}")



