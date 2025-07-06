import React from 'react';
import {
  Box,
  Typography,
  Button,
  Chip,
  Divider,
  Paper,
  Card,
  CardContent,
  CardActions,
  Link,
  Rating
} from '@mui/material';
import { Refresh, OpenInNew, RssFeed, Schedule } from '@mui/icons-material';

interface Article {
  title: string;
  summary: string;
  url: string;
  source: string;
  relevance_score: number;
  published_at: string;
}

interface NewsResponse {
  articles: Article[];
}

interface NewsResultsProps {
  response: NewsResponse;
  onNewRequest: () => void;
}

const NewsResults: React.FC<NewsResultsProps> = ({ response, onNewRequest }) => {
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return 'Recently';
    }
  };

  const getSourceColor = (source: string) => {
    const colors = ['primary', 'secondary', 'success', 'warning', 'info'];
    const index = source.length % colors.length;
    return colors[index] as any;
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
        <RssFeed sx={{ fontSize: '2rem', color: 'primary.main' }} />
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            Your Personalized News
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            {response.articles.length} articles curated for your interests
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Articles List */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {response.articles.map((article, index) => (
          <Card 
            key={index}
            sx={{ 
              display: 'flex',
              flexDirection: 'column',
              '&:hover': {
                boxShadow: 4,
                transform: 'translateY(-2px)',
                transition: 'all 0.3s ease-in-out'
              }
            }}
          >
            <CardContent sx={{ flexGrow: 1 }}>
              {/* Article Header */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" component="h2" sx={{ fontWeight: 600, mb: 1 }}>
                    {article.title}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Chip 
                      label={article.source} 
                      color={getSourceColor(article.source)}
                      size="small"
                    />
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Schedule sx={{ fontSize: 16, color: 'text.secondary' }} />
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(article.published_at)}
                      </Typography>
                    </Box>
                  </Box>
                </Box>
                
                {/* Relevance Score */}
                <Box sx={{ textAlign: 'center', ml: 2 }}>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Relevance
                  </Typography>
                  <Rating 
                    value={article.relevance_score * 5} 
                    readOnly 
                    size="small"
                    precision={0.1}
                  />
                  <Typography variant="caption" color="text.secondary" display="block">
                    {(article.relevance_score * 100).toFixed(0)}%
                  </Typography>
                </Box>
              </Box>

              {/* Article Summary */}
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                {article.summary}
              </Typography>
            </CardContent>

            <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
              <Button
                size="small"
                startIcon={<OpenInNew />}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                sx={{ textTransform: 'none' }}
              >
                Read Full Article
              </Button>
            </CardActions>
          </Card>
        ))}
      </Box>

      {/* No Articles Message */}
      {response.articles.length === 0 && (
        <Paper sx={{ p: 4, textAlign: 'center', bgcolor: 'grey.50' }}>
          <RssFeed sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No articles found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try selecting different interests or adjusting your preferences.
          </Typography>
        </Paper>
      )}

      {/* Action Button */}
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Button
          variant="outlined"
          onClick={onNewRequest}
          startIcon={<Refresh />}
          sx={{ px: 4 }}
        >
          Get New Articles
        </Button>
      </Box>
    </Box>
  );
};

export default NewsResults;
