import os
import time
import json
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def scrape_page(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, 'lxml')

        # Extract title and text
        title = soup.title.string.strip() if soup.title else ''
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        code_blocks = [c.get_text().strip() for c in soup.find_all('code')]
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

        return {
            'url': url,
            'title': title,
            'paragraphs': paragraphs,
            'code_blocks': code_blocks,
            'links': links,
        }
    except Exception as e:
        print(f"[Error] {url}: {e}")
        return None

def crawl_docs(start_url, outdir, max_pages=50, delay=1.0):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'pages'), exist_ok=True)
    visited = set()
    queue = [start_url]
    all_docs = []

    for _ in tqdm(range(max_pages), desc="Crawling pages"):
        if not queue:
            break
        url = queue.pop(0)
        if url in visited or not is_valid_url(url):
            continue
        visited.add(url)

        data = scrape_page(url)
        if not data:
            continue

        # Save each page as a JSON file
        filename = os.path.join(outdir, 'pages', f"{len(visited)}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        all_docs.append(data)

        # Add new links from same domain
        for link in data['links']:
            if urlparse(link).netloc == urlparse(start_url).netloc and link not in visited:
                queue.append(link)

        time.sleep(delay)

    # Save combined JSONL file
    jsonl_path = os.path.join(outdir, 'docs.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n Crawl complete! {len(all_docs)} pages saved to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-url", required=True, help="Starting URL")
    parser.add_argument("--outdir", default="./scraped_docs", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    crawl_docs(args.start_url, args.outdir, args.max_pages, args.delay)

