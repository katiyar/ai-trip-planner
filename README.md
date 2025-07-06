# Personalized News Agent

A fast, intelligent news curation application powered by LangGraph, OpenAI, and Arize observability. Get your top 3 personalized news articles delivered every morning.

## 🚀 Performance Features

- **OpenAI Integration**: Uses GPT-4o-mini for intelligent content analysis and ranking
- **Parallel Processing**: News scraping, content analysis, and ranking run simultaneously
- **Optimized Graph**: Streamlined workflow for efficient news curation
- **LiteLLM Instrumentation**: Comprehensive observability and prompt template tracking

## Architecture

### Frontend (React + TypeScript)
- Modern Material-UI interface
- News interest configuration
- Real-time news digest requests
- Article preview and reading interface

### Backend (FastAPI + LangGraph)
- **Parallel LangGraph Workflow**: 
  - Scraping Node: Multi-source news collection
  - Analysis Node: Content relevance scoring
  - Ranking Node: Personalized article ranking
  - Curation Node: Top 3 article selection with summaries
- **OpenAI LLM**: Intelligent content analysis with `gpt-4o-mini`
- **Comprehensive Tracing**: LangChain + LiteLLM instrumentation

## Quick Start

### 1. Setup Environment

Create a `.env` file in the `backend/` directory:

```bash
# Required: OpenAI API Key (get from https://platform.openai.com)
OPENAI_API_KEY=your_openai_api_key_here

# Required: Arize observability (get from https://app.arize.com)
ARIZE_SPACE_ID=your_arize_space_id
ARIZE_API_KEY=your_arize_api_key

# Optional: For web search capabilities
TAVILY_API_KEY=your_tavily_api_key

# Optional: News API for additional sources
NEWS_API_KEY=your_news_api_key

# LiteLLM Configuration
LITELLM_LOG=DEBUG
```

### 2. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend  
cd ../frontend
npm install
```

### 3. Run the Application

```bash
# Start both services
./start.sh

# Or run separately:
# Backend: cd backend && python main.py
# Frontend: cd frontend && npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Performance Optimizations

### ⚡ OpenAI Integration
- **Fast inference** with GPT-4o-mini for content analysis
- Smart content ranking based on user interests
- 30-second timeout with 2000 max tokens

### 🔄 Parallel Graph Execution
- News scraping, content analysis, and ranking run **simultaneously**
- Reduces total execution time from ~30-60 seconds to ~10-15 seconds
- Final curation waits for all parallel tasks to complete

### 📊 Observability
- **LangChain + LiteLLM instrumentation** for comprehensive tracing
- Prompt template tracking with proper variable separation
- Real-time performance monitoring via Arize platform

## API Endpoints

### POST `/get-news`
Generates personalized news digest.

**Request:**
```json
{
  "interests": ["AI/ML", "Climate Change", "Fintech"],
  "num_articles": 3,
  "sources": ["tech", "business", "science"]
}
```

**Response:**
```json
{
  "articles": [
    {
      "title": "Latest AI Breakthrough in Climate Modeling",
      "summary": "Researchers develop new AI model that predicts climate patterns...",
      "url": "https://example.com/article1",
      "source": "TechCrunch",
      "relevance_score": 0.95,
      "published_at": "2024-01-15T09:00:00Z"
    }
  ]
}
```

### GET `/health`
Health check endpoint.

## Development

### Graph Structure
```
START → [Scraping, Analysis, Ranking] → Curation → END
        (parallel execution)
```

### Key Components
- `scraping_node()`: Multi-source news collection
- `analysis_node()`: Content relevance and quality scoring
- `ranking_node()`: Personalized article ranking based on interests
- `curation_node()`: Top 3 article selection with summaries

### Prompt Templates
All tools use comprehensive prompt templates with proper variable tracking:
- `scraping-v1.0`: Multi-source news collection
- `analysis-v1.0`: Content relevance scoring
- `ranking-v1.0`: Personalized article ranking
- `curation-v1.0`: Final article selection and summarization

## Troubleshooting

### Common Issues
1. **No articles returned**: Check API key configuration in `.env`
2. **Slow responses**: Verify network connectivity for news sources
3. **Graph errors**: Verify all dependencies are installed correctly

### Performance Monitoring
View detailed traces and performance metrics in your Arize dashboard to identify bottlenecks and optimize further.

## Tech Stack

- **Frontend**: React, TypeScript, Material-UI, Axios
- **Backend**: FastAPI, LangGraph, LangChain, OpenAI, LiteLLM
- **Observability**: Arize, OpenInference, OpenTelemetry
- **Infrastructure**: Docker, Docker Compose

## News Sources

The agent aggregates news from:
- RSS feeds from major publications
- Web scraping of news websites
- News APIs (when available)
- Social media trending topics
- Industry-specific publications

## Personalization Features

- **Interest-based filtering**: Focus on topics you care about
- **Learning from feedback**: Improve recommendations over time
- **Customizable delivery**: Choose timing and format
- **Source diversity**: Ensure varied perspectives
- **Quality scoring**: Prioritize authoritative sources
