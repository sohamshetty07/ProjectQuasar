from rss_scraper import get_news_from_rss

def main():
    print("--- Fetching news via RSS Feeds ---")
    
    articles = get_news_from_rss()
    
    if articles:
        print(f"\n--- Top 5 Articles ---")
        for i, article in enumerate(articles[:5]):
            print(f"\n[{i+1}] {article['headline']}")
            print(f"    Source: {article['source']}")
            print(f"    Date: {article['date']}")
            print(f"    Link: {article['link']}")
    else:
        print("\n--- Failed to fetch articles from RSS feeds. ---")

if __name__ == "__main__":
    main()