"""
Copyright Status Verifier

This module provides functions to verify copyright status of content
by searching public records and Gutenberg database.
"""

import time
import json
import os
import requests
from datetime import datetime
from typing import Dict, List, Optional


def search_titles(title_to_search: str) -> Dict:
    """
    Search for copyright records in the U.S. Copyright Office database.
    
    Args:
        title_to_search: Title to search for
        
    Returns:
        Dictionary containing search results
    """
    url = "https://api.publicrecords.copyright.gov/search_service_external/simple_search_dsl"

    params = {
        "page_number": 1,
        "query": title_to_search,
        "column_name": "title",
        "records_per_page": 10,
        "sort_order": "asc",
        "highlight": "true",
        "model": "",
        'registration_class': 'TX',
        'type_of_query': 'starts_with',
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code {response.status_code}")
        return {}


def search_catalog_based_on_title(title: str, api_key: Optional[str] = None) -> Dict:
    """
    Search the National Archives catalog for records.
    
    Args:
        title: Title to search for
        api_key: Optional API key for Archives API
        
    Returns:
        Dictionary containing search results
    """
    url = "https://catalog.archives.gov/api/v2/records/search"

    headers = {
        "Content-Type": "application/json",
    }
    
    if api_key:
        headers["x-api-key"] = api_key
    elif os.getenv("ARCHIVES_API_KEY"):
        headers["x-api-key"] = os.getenv("ARCHIVES_API_KEY")

    params = {
        "title": title
    }

    resp = requests.get(url, headers=headers, params=params)
    return resp.json()


def extract_useful_info(resp: Dict) -> List[Dict]:
    """
    Extract useful information from copyright search results.
    
    Args:
        resp: Response dictionary from copyright search
        
    Returns:
        List of dictionaries containing extracted information
    """
    useful_info_list = []
    for data in resp.get('data', []):
        useful_info = {
            'title': data.get('hit', {}).get('title_concatenated', 'N/A'),
            'recordation_date': data.get('hit', {}).get('recordation_date', 'N/A'),
            'representative_date': data.get('hit', {}).get('representative_date', 'N/A'),
            'execution_date': data.get('hit', {}).get('execution_date', 'N/A'),
            'copyright_number': data.get('hit', {}).get('copyright_number_for_display', 'N/A'),
            'organizations': [org.get('name_organization_indexed_form', 'N/A') for org in
                              data.get('hit', {}).get('organizations', [])],
            'primary_title': {
                'title': data.get('hit', {}).get('primary_titles_list', [{}])[0].get('title_primary_title_title_proper',
                                                                                     'N/A'),
                'statement_of_responsibility': data.get('hit', {}).get('primary_titles_list', [{}])[0].get(
                    'title_primary_title_statement_of_responsibility', 'N/A'),
                'medium': data.get('hit', {}).get('primary_titles_list', [{}])[0].get('title_primary_title_medium',
                                                                                      'N/A')
            },
            'control_number': data.get('hit', {}).get('control_number', 'N/A'),
            'recordation_number': data.get('hit', {}).get('recordation_number', 'N/A'),
            'general_note': data.get('hit', {}).get('general_note', ['N/A'])[0]
        }
        useful_info_list.append(useful_info)
    return useful_info_list


def search_book_gutenberg(title: str, cache_file: Optional[str] = None) -> Dict:
    """
    Search for books in Project Gutenberg database.
    
    Args:
        title: Book title to search for
        cache_file: Optional path to cache file
        
    Returns:
        Dictionary containing book information
    """
    if cache_file is None:
        cache_file = 'cache_search_book_gutenberg.json'
    
    cache = {}

    # Load cache if exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Check cache first
    if title in cache:
        return cache[title]

    # Search Gutenberg API
    base_url = "https://gutendex.com/books"
    params = {
        'search': title
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if results['results']:
            # Find book with highest download count
            top_book = max(results['results'], key=lambda x: x['download_count'])
            book_info = {
                'Title': top_book['title'],
                'Author': ', '.join(author['name'] for author in top_book['authors']),
                'Download count': top_book['download_count'],
                'Copyright status': 'Public domain' if not top_book['copyright'] else 'Protected'
            }
            cache[title] = book_info
        else:
            cache[title] = "No results found."

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=4)

        return cache[title]
    else:
        return "No results found."


def is_public_domain(publication_year: int) -> bool:
    """
    Check if a work is in the public domain based on publication year.
    
    Args:
        publication_year: Year of publication
        
    Returns:
        True if work is in public domain, False otherwise
    """
    current_year = datetime.now().year

    # Works published before 1923 are in public domain
    if publication_year < 1923:
        return True

    # Works published between 1923 and 1977
    elif 1923 <= publication_year <= 1977:
        if publication_year + 95 < current_year:
            return True
        else:
            return False

    # Works published after 1978
    elif publication_year >= 1978:
        if publication_year + 95 < current_year:
            return True
        else:
            return False

    else:
        raise ValueError("Invalid publication year")


def search_book(title: str, cache_file: Optional[str] = None) -> Dict:
    """
    Search for a book, first checking Gutenberg, then public records.
    
    Args:
        title: Book title to search for
        cache_file: Optional cache file path
        
    Returns:
        Dictionary containing book information
    """
    result = search_book_gutenberg(title, cache_file)
    print(f'Search result from Gutenberg for title "{title}": {result}')
    
    if result == "No results found.":
        # Try public records search
        resp = search_titles(title)
        result = extract_useful_info(resp)
        time.sleep(3)  # Rate limiting
    
    return result



