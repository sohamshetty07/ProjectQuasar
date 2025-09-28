import ollama
import json

# This is the master prompt that instructs the model how to behave.
SYSTEM_PROMPT = """
You are a highly specialised financial news analysis engine. Your task is to analyse the provided news article text and return a structured JSON object with the following keys:
- "sentiment_score": A float between -1.0 (very negative) and 1.0 (very positive).
- "sentiment_reasoning": A brief, one-sentence explanation for the sentiment score.
- "key_entities": A list of the most important financial entities mentioned (e.g., companies, products, regulations).
- "predicted_impact": A rating from 1 to 5 on the potential market impact for the main entity, where 1 is minimal and 5 is major.

Do not include any text or formatting outside of the single JSON object.
"""

USER_PROMPT_TEMPLATE = "Analyse this news article: Headline: '{headline}' Summary: '{summary}'"

def analyze_news_with_llm(article):
    """
    Analyzes a single news article using Finance-Llama-8B via Ollama.
    Returns a dictionary of extracted features.
    """
    # Default output in case of an error
    default_output = {
        'sentiment_score': 0.0,
        'sentiment_reasoning': 'Error during analysis.',
        'key_entities': [],
        'predicted_impact': 1
    }
    
    try:
        headline = article.get('headline', '')
        summary = article.get('text', '')
        
        user_prompt = USER_PROMPT_TEMPLATE.format(headline=headline, summary=summary)
        
        # This is the core call to the Ollama service
        response = ollama.chat(
            model='martain7r/finance-llama-8b:q4_k_m', # Specify the exact model version
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0.0} # Use low temperature for consistent JSON output
        )
        
        response_content = response['message']['content']
        # Clean the response to ensure it's valid JSON
        cleaned_json_str = response_content.strip().replace('```json', '').replace('```', '')
        
        parsed_output = json.loads(cleaned_json_str)
        return parsed_output

    except Exception as e:
        print(f"Error processing article '{headline}'. Error: {e}")
        return default_output

# This block allows you to run this file directly to test the function
if __name__ == "__main__":
    print("--- Testing the LLM Intelligence Layer ---")
    
    # This is the same article data your RSS scraper collects
    example_article = {
        'headline': 'Tata Motors smashes profit expectations with record sales', 
        'text': 'The company announced a 50% year-on-year increase in net profit, driven by strong performance in its JLR division.'
    }
    
    analysis_result = analyze_news_with_llm(example_article)
    
    print("\n--- Analysis Result ---")
    # Pretty-print the JSON output
    print(json.dumps(analysis_result, indent=2))