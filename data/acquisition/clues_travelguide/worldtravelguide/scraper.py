import csv
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def retrieve_country_urls():
    """ Extracts all the urls to all country guides from the website https://www.worldtravelguide.net/country-guides/
     and stores them in a .csv file. """
    start_url = "https://www.worldtravelguide.net/country-guides/"
    page = requests.get(start_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # find all the links to the country guides
    links = soup.findAll('a')
    countries = []

    for link in links:
        href = link.get('href')
        if href is not None and href.startswith('/guides/'):
            countries.append(href)

    # create a csv file with the proper urls for each country
    base_url = 'https://www.worldtravelguide.net'
    with open('country_urls.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Continent', 'Country', 'URL'])
        for country_url in countries:
            # Try-except block to handle urls that don't follow the pattern
            try:
                continent, country = country_url.split('/')[2], country_url.split('/')[3]
                writer.writerow([continent, country, base_url + country_url])
            except:
                print('Error with url:', country_url)


def retrieve_country_info():
    """ From a dataframe of urls extract all the information about each country and stores them in a .csv file. """
    # read the csv file with the country urls
    df = pd.read_csv('country_urls.csv')
    data_rows = []

    # for each country, retrieve the information
    for url in tqdm(df['URL']):
        # Extract the continent and country from the url
        splitted_url = url.split('/')
        continent, country = splitted_url[4], splitted_url[5]

        content = [continent, country]
        # Retrieve the page content
        for part in ['', 'history-language-culture/', 'weather-climate-geography/']:
            complete_url = url + part
            page = requests.get(complete_url)
            soup = BeautifulSoup(page.content, 'html.parser')
            # Find the div tag with the specific information needed
            target_div = soup.find('div', {'xmlns:fn': 'http://www.w3.org/2005/xpath-functions', 'itemprop': 'text'})
            if target_div is None:
                target_div = soup.find('article', {'class': 'col-md-7 col-sm-7 main_content'})
            # Extract all paragraph texts
            paragraphs = [p.get_text() for p in target_div.find_all('p')]
            content.append(clean_text(paragraphs))

        data_rows.append(content)

    # Store the information in a csv file
    with open('country_info.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Continent', 'Country', 'General', 'History and Culture', 'Weather and Geography'])

        for row in data_rows:
            writer.writerow(row)


def clean_text(text_list):
    """ Takes a list of strings and cleans them into a combined text string with correct formatting. """
    # Join the list into one string
    joined_text = ' '.join(text_list)
    # Replace newlines with spaces
    joined_text = joined_text.replace('\n', ' ')
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(' +', ' ', joined_text).strip()

    return cleaned_text


def filter(countries: list = ['china', 'india'], columns: list = ['General', 'History and Culture']):
    """ Filtering out data which we don't use for model training.
    Can filter rows of named countries and also full columns. """
    # Load data
    df = pd.read_csv('country_info.csv')
    # filter columns
    df = df.drop(columns=columns)
    # filter rows which belong to china or india
    df = df[~df['Country'].isin(['china', 'india'])]
    # save new dataframe
    df.to_csv(path_or_buf='country_info_filtered.csv', index=False)

    return df


if __name__ == '__main__':
    retrieve_country_urls()
    retrieve_country_info()
    filter()
