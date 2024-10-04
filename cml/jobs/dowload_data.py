from bs4 import BeautifulSoup, SoupStrainer
import requests

WEB_ROOT = "https://ktendering.com.kw/esop/kuw-kpc-host/public/"
SAVE_FOLDER = "data/pdf/"


def download_file(file_link, folder):
    file = requests.get(file_link).content
    name = file_link.split("/")[-1]
    save_path = folder + name

    print("Saving file:", save_path)
    with open(save_path, "wb") as fp:
        fp.write(file)


def download_job():
    scrape_url = "https://ktendering.com.kw/esop/kuw-kpc-host/public/ktendering/web/helpful_documents_guidelines.html"
    r = requests.get(scrape_url)

    links = []
    for link in BeautifulSoup(r.content, "html.parser", parse_only=SoupStrainer("a")):
        if link.has_attr("href"):
            links.append(WEB_ROOT + link.attrs["href"].removeprefix("../../").strip())

    pdf_links = set(filter(lambda e: e.endswith(".pdf"), links))
    for link in list(pdf_links):
        download_file(link, SAVE_FOLDER)
