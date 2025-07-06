import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Card,
  CardContent,
  Chip
} from '@mui/material';
import { RssFeed, Article } from '@mui/icons-material';
import NewsConfigForm from './components/NewsConfigForm';
import NewsResults from './components/NewsResults';

// Define types for news functionality
interface NewsRequest {
  interests: string[];
  num_articles: number;
  sources?: string[];
}

interface ArticleType {
  title: string;
  summary: string;
  url: string;
  source: string;
  relevance_score: number;
  published_at: string;
}

interface NewsResponse {
  articles: ArticleType[];
}

const theme = createTheme({
  palette: {
    primary: {
      main: '#1565c0', // News blue
    },
    secondary: {
      main: '#ef6c00', // Orange accent
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h3: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
});

function App() {
  const [newsResponse, setNewsResponse] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGetNews = async (newsRequest: NewsRequest) => {
    setLoading(true);
    setError(null);
    setNewsResponse(null);

    try {
      const response = await fetch('http://localhost:8000/get-news', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newsRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: NewsResponse = await response.json();
      setNewsResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching news:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewRequest = () => {
    setNewsResponse(null);
    setError(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <RssFeed sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Personalized News Agent
            </Typography>
            <Article />
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          {/* Hero Section */}
          <Paper
            sx={{
              p: 4,
              mb: 4,
              background: 'linear-gradient(135deg, #1565c0 0%, #0d47a1 100%)',
              color: 'white'
            }}
          >
            <Typography variant="h3" component="h1" gutterBottom align="center">
              Your Personalized News Feed
            </Typography>
            <Typography variant="h6" align="center" sx={{ opacity: 0.9 }}>
              Get the top 3 articles curated just for your interests every day.
              Stay informed without the noise.
            </Typography>
          </Paper>

          {/* Features Cards */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3, mb: 4 }}>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🔍 Smart Curation
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    AI-powered filtering finds the most relevant articles for your interests
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📊 Relevance Scoring
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Each article is scored for relevance to your specific interests
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🌐 Multi-Source
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Aggregates from multiple reputable news sources and publications
                  </Typography>
                </CardContent>
              </Card>
            </Box>
            <Box sx={{ flex: '1 1 250px', minWidth: '250px' }}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 2 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ⚡ Real-time
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Fresh articles updated throughout the day as news breaks
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Box>

          {/* Main Content */}
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
            <Box sx={{ flex: '0 0 auto', width: { xs: '100%', md: '400px' } }}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom>
                  Configure Your News
                </Typography>
                <NewsConfigForm onSubmit={handleGetNews} loading={loading} />
              </Paper>
            </Box>

            <Box sx={{ flex: 1 }}>
              {error && (
                <Paper sx={{ p: 3, mb: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
                  <Typography variant="h6" gutterBottom>
                    Error
                  </Typography>
                  <Typography>{error}</Typography>
                </Paper>
              )}

              {newsResponse && (
                <Paper sx={{ p: 3 }}>
                  <NewsResults response={newsResponse} onNewRequest={handleNewRequest} />
                </Paper>
              )}

              {!newsResponse && !loading && !error && (
                <Paper sx={{ p: 6, textAlign: 'center', bgcolor: 'grey.50' }}>
                  <RssFeed sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary">
                    Select your interests to get personalized news
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Our AI will curate the most relevant articles just for you
                  </Typography>
                </Paper>
              )}
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
