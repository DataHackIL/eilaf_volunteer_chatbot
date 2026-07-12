import os
from pathlib import Path

from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR


class BaseScraper:
    def __init__(self, target_url: str, static_dir: os.PathLike = DEFAULT_STATIC_DIR):
        self.target_url = target_url
        self.static_dir = Path(static_dir)
        self._extension = 'txt'

    def scrape(self) -> str:
        """Scrape the target URL and return the scraped content."""
        raise NotImplementedError

    def format(self, content: str) -> str:
        """Format the scraped content for storage."""
        raise NotImplementedError
    
    def save(self, content: str, filename: str) -> None:
        """Save the formatted content to a file in the static directory."""
        self.static_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.static_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def __call__(self) -> None:
        """Scrape, format, and save the content."""
        content = self.scrape()
        formatted_content = self.format(content)
        filename = f"{self.target_url.replace('/', '_')}.{self._extension}"
        self.save(formatted_content, filename)


class BaseHTMLScraper(BaseScraper):
    def scrape(self) -> str:
        """Scrape the target URL using HTTP GET and return the response text."""
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }


        response = requests.get(self.target_url, headers=headers)
        response.raise_for_status()
        return response.text
    
if __name__ == "__main__":
    # Example usage: scrape a URL and save the content to the static directory.
    scraper = BaseHTMLScraper("https://www.nevo.co.il/law_html/law00/71835.htm")
    scraper()