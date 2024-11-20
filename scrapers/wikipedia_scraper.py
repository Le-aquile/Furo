import wikipediaapi

def get_pages_in_category(category_name, lang='it'):
    """
    Raccoglie le pagine di una categoria su Wikipedia.
    Args:
        category_name (str): Nome della categoria (senza "Categoria:").
        lang (str): Codice della lingua (default: italiano - 'it').

    Returns:
        List: Titoli delle pagine all'interno della categoria.
    """
    # Specifica un user agent personalizzato
    user_agent = "WikipediaInfoGetter/1.0"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)
    page_content = []
    # Accedi alla categoria
    category = wiki_wiki.page(f"Categoria:{category_name}")
    if not category.exists():
        print(f"La categoria '{category_name}' non esiste.")
        return []
    
    print(f"Pagine nella categoria '{category_name}':")
    page_titles = []
    for subpage in category.categorymembers.values():
        if subpage.ns == 0:  # Verifica se è una pagina di contenuto (non sottocategoria, ecc.)
            print(f"- {subpage.title}")
            page_titles.append(subpage.title)
            page_content.append(subpage.text)
    
    return zip(page_titles, page_content)


if __name__ == "__main__":
    # Esempio di utilizzo
    category_name = "Intelligenza artificiale"
    pages = get_pages_in_category(category_name)

    # Puoi salvare i titoli o elaborarli ulteriormente
    print("\nElenco di pagine:")
    for page in pages:
        print(page)

