import requests
from bs4 import BeautifulSoup
import time
import os


BASE = "http://piano-midi.de/"

SAVE = os.path.join("midiFiles", "piano-midi")
os.makedirs(SAVE, exist_ok=True)

composer_pages = [
    "bach.htm",
    "beeth.htm",
    "chopin.htm"
    "haydn.htm"
    "schub.htm"
    "tschai.htm"
    "brahms.htm"
    "liszt.htm"
    "ravel.htm"
    "rach.htm"
    "schum.htm"
]

def scrape():
    for page in composer_pages:
        print("Scraping:", page)
        url = BASE + page
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a"):
            href = a.get("href", "")
            print("href: ",href)
            if href.endswith(".mid"):
                file_url = BASE + href

                filename = os.path.join(
                    SAVE,
                    href.split("/")[-1]
                )

                r = requests.get(file_url)

                with open(filename, "wb") as f:
                    f.write(r.content)

                print("downloaded:", filename)

                time.sleep(0.5)  # don't spam server


if __name__ == "__main__":
    scrape()