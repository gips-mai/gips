import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import json


def retrieve_country_urls():
    start_url = "https://culturalatlas.sbs.com.au/countries"
    page = requests.get(start_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # find all the links to the country guides
    links = soup.findAll('a')
    countries = []
    base_url = 'https://culturalatlas.sbs.com.au'

    # filter the links to only include the country guides
    for link in links:
        href = link.get('href')
        if href is not None and href.endswith('-culture'):
            # assemble the full url
            countries.append(base_url + href)

    # store the urls in a csv file
    with open('country_urls.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['COUNTRY', 'URL'])
        for country_url in countries:
            country_name = country_url.split('/')[-1].replace('-culture', '')
            writer.writerow([country_name, create_core_concepts_url(country_url)])


def create_core_concepts_url(country_url):
    # create the url for the core concepts page
    # e.g. from  https://culturalatlas.sbs.com.au/egyptian-culture to
    # https://culturalatlas.sbs.com.au/egyptian-culture/egyptian-culture-core-concepts#egyptian-culture-core-concepts

    return country_url + '/' + country_url.split('/')[-1] + '-core-concepts#' + country_url.split('/')[
        -1] + '-core-concepts'


def retrieve_country_info():
    # read the csv file with the country urls
    df = pd.read_csv('country_urls.csv')
    data_rows = []

    # for each country, retrieve the information
    i = 0
    for url in tqdm(df['URL']):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        # Find the div tag with the specific information needed
        try:
            target_div = soup.find('div', {'class': 'text-content'})
            data_react_props = target_div.find('div', {'data-react-class': 'textcontent/TextContentRenderer'}).get(
                'data-react-props')
            data_dict = json.loads(data_react_props)  # convert the string representation to an actual dictionary
        except AttributeError:
            print(f'Error with {df["COUNTRY"][i]}')
            continue

        # extract the information
        country_text = []
        for entry in data_dict['value']:
            for p in entry['children']:
                if 'text' in p:
                    country_text.append(p['text'])

        country_name = df['COUNTRY'][i]
        data_rows.append([country_name, country_text])
        i += 1

    # store the information in a csv file
    with open('country_info.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Country', 'Info'])
        for row in data_rows:
            writer.writerow(row)

retrieve_country_info()
