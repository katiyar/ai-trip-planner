import React, { useState } from 'react';
import {
  TextField,
  Button,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  SelectChangeEvent,
  Chip,
  Typography,
  Slider,
  FormControlLabel,
  Checkbox,
  FormGroup
} from '@mui/material';
import { RssFeed } from '@mui/icons-material';

interface NewsRequest {
  interests: string[];
  num_articles: number;
  sources?: string[];
}

interface NewsConfigFormProps {
  onSubmit: (newsRequest: NewsRequest) => void;
  loading: boolean;
}

const NewsConfigForm: React.FC<NewsConfigFormProps> = ({ onSubmit, loading }) => {
  const [selectedInterests, setSelectedInterests] = useState<string[]>([]);
  const [numArticles, setNumArticles] = useState<number>(3);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);

  const availableInterests = [
    'AI/Machine Learning',
    'Technology',
    'Science',
    'Business',
    'Finance',
    'Climate Change',
    'Healthcare',
    'Politics',
    'Sports',
    'Entertainment',
    'Cybersecurity',
    'Startups',
    'Cryptocurrency',
    'Space',
    'Energy'
  ];

  const availableSources = [
    'tech',
    'business',
    'science',
    'general',
    'health'
  ];

  const handleInterestChange = (interest: string) => {
    setSelectedInterests(prev => 
      prev.includes(interest) 
        ? prev.filter(i => i !== interest)
        : [...prev, interest]
    );
  };

  const handleSourceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const source = event.target.value;
    setSelectedSources(prev => 
      prev.includes(source) 
        ? prev.filter(s => s !== source)
        : [...prev, source]
    );
  };

  const handleNumArticlesChange = (event: Event, newValue: number | number[]) => {
    setNumArticles(newValue as number);
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (selectedInterests.length > 0) {
      onSubmit({
        interests: selectedInterests,
        num_articles: numArticles,
        sources: selectedSources.length > 0 ? selectedSources : undefined
      });
    }
  };

  const isFormValid = selectedInterests.length > 0;

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Select Your Interests
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        {availableInterests.map((interest) => (
          <Chip
            key={interest}
            label={interest}
            onClick={() => handleInterestChange(interest)}
            color={selectedInterests.includes(interest) ? "primary" : "default"}
            variant={selectedInterests.includes(interest) ? "filled" : "outlined"}
            sx={{ m: 0.5 }}
            disabled={loading}
          />
        ))}
      </Box>

      <Typography variant="h6" gutterBottom>
        Number of Articles: {numArticles}
      </Typography>
      
      <Slider
        value={numArticles}
        onChange={handleNumArticlesChange}
        min={1}
        max={10}
        marks
        valueLabelDisplay="auto"
        disabled={loading}
        sx={{ mb: 3 }}
      />

      <Typography variant="h6" gutterBottom>
        News Sources (Optional)
      </Typography>
      
      <FormGroup row sx={{ mb: 3 }}>
        {availableSources.map((source) => (
          <FormControlLabel
            key={source}
            control={
              <Checkbox
                value={source}
                checked={selectedSources.includes(source)}
                onChange={handleSourceChange}
                disabled={loading}
              />
            }
            label={source.charAt(0).toUpperCase() + source.slice(1)}
          />
        ))}
      </FormGroup>

      <Button
        type="submit"
        fullWidth
        variant="contained"
        disabled={!isFormValid || loading}
        startIcon={loading ? <CircularProgress size={20} /> : <RssFeed />}
        sx={{ mt: 3, mb: 2, py: 1.5 }}
      >
        {loading ? 'Curating Your News...' : 'Get My News'}
      </Button>

      {selectedInterests.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="textSecondary">
            Selected interests: {selectedInterests.join(', ')}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default NewsConfigForm;
