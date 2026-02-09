import os
from fastapi import FastAPI, HTTPException, File, Response
from dotenv import load_dotenv
from pathlib import Path

from models import NewsRequest

from news_scraper import NewsScraper

from utils import (
    text_to_audio_elevenlabs_sdk,
    generate_broadcast_news
)

app = FastAPI()
load_dotenv()


@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        results = {}

        if request.source_type in ["news", "both"]:
            # scrape google news
            news_scraper = NewsScraper()
            results["news"] = await news_scraper.scrape_news(request.topics)

        news_data = results.get("news", {})

        # setup LLM summarizer
        news_summary = generate_broadcast_news(
            api_key=os.getenv("GROQ_API_KEY"),
            news_data=news_data,
            topics=request.topics
            )

        # Convert summary to audio
        audio_path = text_to_audio_elevenlabs_sdk(
            text=news_summary,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            output_dir="audio"
        )

        if audio_path and Path(audio_path).exists():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=news-summary.mp3"}
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=1234,
        reload=True
    )