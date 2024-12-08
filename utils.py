"""
Utility functions for Sue AI Assistant
"""
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import lru_cache
import aiohttp
from bs4 import BeautifulSoup
import html2text
from googlesearch import search
from config import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class WebCache:
    """Simple cache with TTL"""
    def __init__(self, ttl: int = config.CACHE_TTL, max_size: int = config.WEB_CACHE_SIZE):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        if datetime.now() - item["timestamp"] > timedelta(seconds=self.ttl):
            del self._cache[key]
            return None
            
        return item["value"]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        if len(self._cache) >= self.max_size:
            # Remove oldest item
            oldest = min(self._cache.items(), key=lambda x: x[1]["timestamp"])
            del self._cache[oldest[0]]
            
        self._cache[key] = {
            "value": value,
            "timestamp": datetime.now()
        }

# Create global cache instance
web_cache = WebCache()

async def process_webpage_async(url: str) -> str:
    """Process webpage content asynchronously"""
    try:
        # Check cache first
        cached_content = web_cache.get(url)
        if cached_content:
            logger.info(f"Using cached content for {url}")
            return cached_content

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": config.USER_AGENT}
            ) as response:
                if response.status != 200:
                    return f"Ошибка: статус {response.status}"
                
                html = await response.text()
                
                # Extract title for logging
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.title.string if soup.title else "No title"
                logger.info(f"Page title: {title}")

                # Convert to markdown
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.ignore_emphasis = False  # Возможно, стоит сохранить акценты
                markdown = h.handle(html).strip()
                
                if not markdown:
                    return "Не удалось извлечь контент"
                
                # Cache the result
                web_cache.set(url, markdown)
                logger.info(f"Successfully processed and cached content from {url}")
                
                return markdown

    except asyncio.TimeoutError:
        return "Ошибка: превышено время ожидания"
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return f"Ошибка при обработке страницы: {str(e)}"

def clean_search_query(query: str) -> str:
    """Clean and prepare search query"""
    # Remove quotes
    query = query.replace('"', '').replace('"', '').replace('"', '')
    # Remove question marks at the end
    query = query.rstrip('?')
    # Remove search command words
    for word in ['найди', 'поищи', 'расскажи', 'что такое']:
        if query.lower().startswith(word):
            query = query[len(word):].lstrip()
    return query.strip()

async def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using Google Search"""
    try:
        # Clean query
        clean_query = clean_search_query(query)
        logger.info(f"Cleaned query: {clean_query}")

        # Check cache first
        cache_key = f"search_{clean_query}_{num_results}"
        cached_results = web_cache.get(cache_key)
        if cached_results:
            logger.info(f"Using cached search results for query: {clean_query}")
            return cached_results

        results = []
        try:
            # Perform search with Russian language preference
            search_results = search(
                clean_query,
                lang='ru',
                num_results=num_results,
                advanced=True
            )
            
            for r in search_results:
                results.append({
                    'title': r.title if hasattr(r, 'title') and r.title else r.url,
                    'link': r.url,
                    'snippet': r.description if hasattr(r, 'description') and r.description else "Описание недоступно"
                })
                
        except Exception as e:
            logger.warning(f"Initial search error: {str(e)}")
            
            # Try with fewer keywords if initial search fails
            if not results:
                keywords = ' '.join(clean_query.split()[:3])
                logger.info(f"Retrying with keywords: {keywords}")
                try:
                    search_results = search(
                        keywords,
                        lang='ru',
                        num_results=num_results,
                        advanced=True
                    )
                    
                    for r in search_results:
                        results.append({
                            'title': r.title if hasattr(r, 'title') and r.title else r.url,
                            'link': r.url,
                            'snippet': r.description if hasattr(r, 'description') and r.description else "Описание недоступно"
                        })
                except Exception as e:
                    logger.warning(f"Retry search error: {str(e)}")

        if not results:
            logger.warning(f"No results found for query: {clean_query}")
            return []

        # Cache results
        web_cache.set(cache_key, results)
        logger.info(f"Found {len(results)} results for query: {clean_query}")
        return results

    except Exception as e:
        logger.error(f"Search error for query '{query}': {str(e)}")
        return []
