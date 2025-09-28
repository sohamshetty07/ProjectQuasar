import feedparser

# A list of high-quality Indian financial news RSS feeds
RSS_FEEDS = {
    "Livemint - Markets": "https://www.livemint.com/rss/markets",
    "Livemint - Companies": "https://www.livemint.com/rss/companies",
    "Economic Times - Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
}

def get_news_from_rss():
    """
    Parses multiple RSS feeds and returns a consolidated list of articles.
    """
    articles = []
    print("[RSS Scraper] Fetching news from RSS feeds...")
    
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            # The 'entries' key contains the list of articles
            for entry in feed.entries:
                articles.append({
                    'headline': entry.title,
                    'link': entry.link,
                    'source': source_name,
                    'date': entry.get('published', 'N/A'),
                    'text': entry.get('summary', 'N/A')
                })
        except Exception as e:
            print(f"[RSS Scraper] Could not parse feed {source_name}. Error: {e}")
            
    print(f"[RSS Scraper] Successfully fetched {len(articles)} articles.")
    return articles