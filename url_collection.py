from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, urlunparse

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument(
   "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
service = Service()
driver = webdriver.Chrome(service=service, options=chrome_options)

def normalize_url(url):
    parsed = urlparse(url)
    parsed = parsed._replace(fragment='')
    path = parsed.path if parsed.path != '/' else ''
    if path.endswith('/'):
        path = path.rstrip('/')
    parsed = parsed._replace(path=path)
    normalized = urlunparse(parsed)
    return normalized

to_visit = {"https://seedfund.startupindia.gov.in"}
visited = set()

try:
    while to_visit:
        url = to_visit.pop()
        url = normalize_url(url)
        if url in visited:
            continue
        visited.add(url)

        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            links = driver.find_elements(By.TAG_NAME, "a")
            for link in links:
                href = link.get_attribute("href")
                if href and href.startswith("https") and "seedfund.startupindia.gov.in" in href:
                    href = normalize_url(href)
                    if href not in visited and href not in to_visit:
                        to_visit.add(href)

        except Exception as e:
            print("Error with", url, ":", e)

finally:
    with open('urls.txt', 'w') as f:
            for i in visited:
                f.write(i+"\n");
    driver.quit()

print("\nTotal unique URLs found:", len(visited))
for link in sorted(visited):
    print(link)
