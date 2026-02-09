import os
import ssl
import requests
import feedparser
from datetime import datetime
from elevenlabs import ElevenLabs
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
load_dotenv()

# Bypass SSL
ssl._create_default_https_context = ssl._create_unverified_context

def generate_google_news_url(keyword: str) -> str:
    # RSS URL format
    rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    return rss_url


def generate_news_urls_to_scrape(list_of_keywords):
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword] = generate_google_news_url(keyword)
    return valid_urls_dict


def scrape_google_news(url):
    # Fetch the content with a header to avoid being flagged as a basic bot
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)

    # Parse the text content of the response
    feed = feedparser.parse(response.text)

    headlines = []
    if feed.bozo == 0:
        for entry in feed.entries:
            headlines.append(entry.title)
    else:
        print("Error parsing feed. Google might be blocking the request.")

    return "\n".join(headlines)


def summarize_with_groq_news_script(api_key: str, headlines: str) -> str:
    """
    Summarizes scraped headlines using Groq's Llama-3.1-8b-instant model.
    """
    system_prompt = """
    You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.

    The final output will be read aloud by a news anchor or text-to-speech engine. So:
    - Do not include any special characters, emojis, formatting symbols, or markdown.
    - Do not add any preamble or framing like "Here's your summary" or "Let me explain".
    - Write in full, clear, spoken-language paragraphs.
    - Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
    - Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
    - Start right away with the actual script, using transitions between topics if needed.

    Remember: Your only output should be a clean script that is ready to be read out loud.
    """

    try:
        # Initializing Groq instead of Anthropic
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.4,
            max_tokens=1000
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=headlines)
        ])

        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

def generate_broadcast_news(api_key, news_data, topics):
    """
    Generates broadcast news using Groq's Llama-3.3-70b-versatile model.
    """
    system_prompt = """
        You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

        For each topic, STRUCTURE BASED ON AVAILABLE DATA:
        1. If news exists: "According to official reports..." + summary
        2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
        3. If both exist: Present news first, then Reddit reactions
        4. If neither exists: Skip the topic (shouldn't happen)

        Formatting rules:
        - ALWAYS start directly with the content, NO INTRODUCTIONS
        - Keep audio length 60-120 seconds per topic
        - Use natural speech transitions like "Meanwhile, online discussions..." 
        - Incorporate 1-2 short quotes from Reddit when available
        - Maintain neutral tone but highlight key sentiments
        - End with "To wrap up this segment..." summary

        Write in full paragraphs optimized for speech synthesis. Avoid markdown.
    """

    try:
        topic_blocks = []
        for topic in topics:
            news_content = news_data["news_analysis"].get(topic) if news_data else ''
            context = []
            if news_content:
                context.append(f"OFFICIAL NEWS CONTENT:\n{news_content}")
            
            if context:
                topic_blocks.append(
                    f"TOPIC: {topic}\n\n" +
                    "\n\n--- NEW TOPIC ---\n\n".join(context)
                )

        user_prompt = (
            "Create broadcast segments for these topics using available sources:\n\n" +
            "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
        )

        # Initializing Groq for the larger generation task
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.3,
            max_tokens=4000,
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content

    except Exception as e:
        raise e
    

def text_to_audio_elevenlabs_sdk(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    output_dir: str = "audio",
    api_key: str = None
) -> str:
    """
    Converts text to speech using ElevenLabs SDK and saves it to audio/ directory.

    Returns:
        str: Path to the saved audio file.
    """
    try:
        api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")

        # Initialize client
        client = ElevenLabs(api_key=api_key)

        # Get the audio generator
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)

        # Write audio chunks to file
        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return filepath

    except Exception as e:
        raise e






# url = generate_google_news_url("bitcoin")
# scraped_headlines = scrape_google_news(url)
# summarize_with_anthropic_news_script(scraped_headlines)