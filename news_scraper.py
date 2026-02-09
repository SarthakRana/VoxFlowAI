import os
import asyncio
from typing import List, Dict
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()
from utils import (
    generate_news_urls_to_scrape,
    scrape_google_news,
    summarize_with_groq_news_script,
    )


class NewsScraper:
    _rate_limiter = AsyncLimiter(5, 1)  # 5 requests/second

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        """Scrape and analyze news articles"""
        results = {}
        
        for topic in topics:
            async with self._rate_limiter:
                try:
                    urls = generate_news_urls_to_scrape([topic])
                    headlines = scrape_google_news(urls[topic])
                    summary = summarize_with_groq_news_script(
                        api_key=os.getenv("GROQ_API_KEY"),
                        headlines=headlines
                    )
                    results[topic] = summary
                except Exception as e:
                    results[topic] = f"Error: {str(e)}"
                await asyncio.sleep(1)  # Avoid overwhelming news sites

        return {"news_analysis" : results}