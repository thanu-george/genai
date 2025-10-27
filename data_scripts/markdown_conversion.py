from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import markdownify
import os
from bs4 import BeautifulSoup


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(root_data_dir, exist_ok=True)

rendered_html = os.path.join(root_data_dir, "rendered_html")
first_clean = os.path.join(root_data_dir, "first_clean")

os.makedirs(rendered_html, exist_ok=True)
os.makedirs(first_clean, exist_ok=True)

def cleanup(content):
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer","img","video","iframe"]):
        tag.decompose()
    return str(soup)

with open("urls.csv", 'r') as f:
    urls = [line.strip().replace("\ufeff", "") for line in f if line.strip()]

    try:
        for i, url in enumerate(urls):
            print(f"Processing {i+1}/{len(urls)}: {url}")
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                html_content = driver.page_source
                filename = url.replace("https://", "").replace("http://", "").replace("/", "_")

                raw_filepath = os.path.join(rendered_html, filename + ".html")
                with open(raw_filepath, "w", encoding="utf-8") as f:
                    f.write(html_content)

                clean_html=cleanup(html_content)
                markdown_content = markdownify.markdownify(clean_html)
                
                md_filename = filename + ".md"
                md_filepath = os.path.join(first_clean, filename + ".md")
                with open(md_filepath, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

            except Exception as e:
                print(f"Error processing {url}: {e}")
    finally:
        driver.quit()

