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
chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)


os.makedirs("data", exist_ok=True)

def cleanup(content):
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer","img","video","iframe"]):
        tag.decompose()
    return str(soup)

with open("urls.txt", 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

    try:
        for i, url in enumerate(urls):
            print(f"Processing {i+1}/{len(urls)}: {url}")
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                html_content = driver.page_source
                clean_html=cleanup(html_content)
                markdown_content = markdownify.markdownify(clean_html)
                
                filename = url.replace("https://", "").replace("http://", "").replace("/", "_")
                if not filename.endswith(".md"):
                    filename += ".md"

                # Save markdown to file
                filepath = os.path.join("data", filename)
                with open(filepath, "w") as md_file:
                    md_file.write(markdown_content)

            except Exception as e:
                print(f"Error processing {url}: {e}")

    finally:
        driver.quit()

