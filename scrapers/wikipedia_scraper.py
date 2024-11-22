import wikipediaapi

def get_pages_in_category(category_name, lang='en', to_zip=False):
    """
    Collects the pages of a category on Wikipedia.
    Args:
        category_name (str): Name of the category (without "Category:").
        lang (str): Language code (default: English - 'en').
        to_zip (bool): If True, returns a list of tuples (page_title, page_content).

    Returns:
        List: Page contents within the category or a list of tuples (page_title, page_content) if to_zip=True.
    """
    # Specify a custom user agent
    user_agent = "WikipediaInfoGetter/1.0"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)
    page_content = []
    # Access the category
    category = wiki_wiki.page(f"Category:{category_name}")
    if not category.exists():
        print(f"Category '{category_name}' does not exist.")
        return []
    
    print(f"Pages in category '{category_name}':")
    page_titles = []
    for subpage in category.categorymembers.values():
        if subpage.ns == 0:  # Check if it is a content page (not a subcategory, etc.)
            print(f"- {subpage.title}")
            page_titles.append(subpage.title)
            page_content.append(subpage.text)
    
    if to_zip:
        return list(zip(page_titles, page_content))
    else:
        return page_content

if __name__ == "__main__":
    # Example of usage
    category_name = "Artificial intelligence"
    pages = get_pages_in_category(category_name)

    # You can save the titles or process them further
    print("\nList of pages:")
    for page in pages:
        print(page)

