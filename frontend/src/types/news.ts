export interface Article {
  title: string;
  summary: string;
  url: string;
  source: string;
  relevance_score: number;
  published_at: string;
}

export interface NewsRequest {
  interests: string[];
  num_articles?: number;
  sources?: string[];
}

export interface NewsResponse {
  articles: Article[];
}

export interface NewsFormData {
  interests: string[];
  numArticles: number;
  selectedSources: string[];
} 