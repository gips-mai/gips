## Data Acquisition

### Downloading datasets from related work

We utilized two datasets from related work:

- Clues from G^3
    - In their paper, the authors utilized a set of clues from a geoguessr guidebook as additional text input to a model
      to classify the correct country
    - We also utilized this data, and downloaded the data from
      the [G^3 repository](https://github.com/g-luo/geolocation_via_guidebook_grounding), which links a guidebook.json
      file to [here](http://geolocation_via_guidebook_grounding.berkeleyvision.org/dataset/)
- OSV5m data:
    - This dataset provides us with image-coordinate pairs, which we train the model on to predict the geolocation of an
      image
    - For downloading the dataset we followed the instructions provided in
      the [OSV5m repository](https://github.com/gastruc/osv5m) wit the `python scripts/download-dataset.py` script.

### Scraping the World Travel Guide

The general idea is to combine the new coordinate based geolocation prediction approach of the OSV5M paper with the
multimodality of the G^3 paper.
In G^3 the authors use a set of clues from a geoguessr guidebook as a static input which combined with the encoded image
via a separate attention module.

We want to additionally enrich these clues with additional information from travel guides to give the model broader
opportunity to learn how to combine clues and image information for the best prediction possible.
To this end, we scraped cultural information, as well as geographic and climate information from various sources.
But after some analysis and consultation from Prof. Rohrbach, we decided to only use the geographic and climate
information, as cultural information is not directly related to images and would most likely confuse the model instead
of doing any good.

So finally we only used the information scraped in the worldtravelguide directory.

The script can be executed by running the following command:

```bash
python clues_travelguide/worldtravelguide/scraper.py
```

