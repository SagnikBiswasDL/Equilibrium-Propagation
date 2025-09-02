import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Box,
  LinearProgress,
  Alert,
  Chip,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Science,
  TrendingUp,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import axios from 'axios';

const Training = () => {
  const [epParams, setEpParams] = useState({
    beta: 0.1,
    learning_rate: 0.001,
    num_epochs: 10,
    hidden_sizes: [500, 500],
    device: 'cpu',
  });

  const [backpropParams, setBackpropParams] = useState({
    learning_rate: 0.001,
    num_epochs: 10,
    hidden_sizes: [500, 500],
    device: 'cpu',
    optimizer: 'adam',
  });

  const [status, setStatus] = useState({
    ep_training: false,
    backprop_training: false,
    ep_progress: 0,
    backprop_progress: 0,
    ep_results: null,
    backprop_results: null,
  });

  const [epHistory, setEpHistory] = useState(null);
  const [backpropHistory, setBackpropHistory] = useState(null);

  // Poll for status updates
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get('/api/status');
        setStatus(response.data);
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const startEpTraining = async () => {
    try {
      await axios.post('/api/train/ep', epParams);
    } catch (error) {
      console.error('Error starting EP training:', error);
    }
  };

  const startBackpropTraining = async () => {
    try {
      await axios.post('/api/train/backprop', backpropParams);
    } catch (error) {
      console.error('Error starting backprop training:', error);
    }
  };

  const stopEpTraining = async () => {
    try {
      await axios.post('/api/stop/ep');
    } catch (error) {
      console.error('Error stopping EP training:', error);
    }
  };

  const stopBackpropTraining = async () => {
    try {
      await axios.post('/api/stop/backprop');
    } catch (error) {
      console.error('Error stopping backprop training:', error);
    }
  };

  const fetchResults = async () => {
    try {
      if (status.ep_results) {
        const response = await axios.get('/api/results/ep');
        setEpHistory(response.data);
      }
      if (status.backprop_results) {
        const response = await axios.get('/api/results/backprop');
        setBackpropHistory(response.data);
      }
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  useEffect(() => {
    if (status.ep_results && status.backprop_results) {
      fetchResults();
    }
  }, [status.ep_results, status.backprop_results]);

  const renderTrainingForm = (type, params, setParams, onStart, onStop, isTraining) => {
    const isEp = type === 'ep';
    
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {isEp ? <Science color="primary" /> : <TrendingUp color="secondary" />}
            <Typography variant="h6" sx={{ ml: 1 }}>
              {isEp ? 'Equilibrium Propagation' : 'Backpropagation'} Training
            </Typography>
          </Box>

          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Learning Rate"
                type="number"
                value={params.learning_rate}
                onChange={(e) => setParams({ ...params, learning_rate: parseFloat(e.target.value) })}
                inputProps={{ step: 0.0001, min: 0.0001, max: 0.1 }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Number of Epochs"
                type="number"
                value={params.num_epochs}
                onChange={(e) => setParams({ ...params, num_epochs: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 100 }}
              />
            </Grid>
            {isEp && (
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Beta (EP Parameter)"
                  type="number"
                  value={params.beta}
                  onChange={(e) => setParams({ ...params, beta: parseFloat(e.target.value) })}
                  inputProps={{ step: 0.01, min: 0.01, max: 1.0 }}
                />
              </Grid>
            )}
            {!isEp && (
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Optimizer</InputLabel>
                  <Select
                    value={params.optimizer}
                    label="Optimizer"
                    onChange={(e) => setParams({ ...params, optimizer: e.target.value })}
                  >
                    <MenuItem value="adam">Adam</MenuItem>
                    <MenuItem value="sgd">SGD</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            )}
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Device</InputLabel>
                <Select
                  value={params.device}
                  label="Device"
                  onChange={(e) => setParams({ ...params, device: e.target.value })}
                >
                  <MenuItem value="cpu">CPU</MenuItem>
                  <MenuItem value="cuda">CUDA (if available)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<PlayArrow />}
              onClick={onStart}
              disabled={isTraining}
              fullWidth
            >
              Start Training
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<Stop />}
              onClick={onStop}
              disabled={!isTraining}
              fullWidth
            >
              Stop Training
            </Button>
          </Box>

          {isTraining && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress 
                variant="determinate" 
                value={isEp ? status.ep_progress : status.backprop_progress} 
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Progress: {isEp ? status.ep_progress : status.backprop_progress}%
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderResults = (type, history) => {
    if (!history) return null;

    const isEp = type === 'ep';
    const title = isEp ? 'EP Training Results' : 'Backprop Training Results';

    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          
          {history.error ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {history.error}
            </Alert>
          ) : (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Final Training Accuracy
                </Typography>
                <Typography variant="h6" color="primary">
                  {history.train_accuracies?.[history.train_accuracies.length - 1]?.toFixed(2)}%
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Final Validation Accuracy
                </Typography>
                <Typography variant="h6" color="secondary">
                  {history.val_accuracies?.[history.val_accuracies.length - 1]?.toFixed(2)}%
                </Typography>
              </Grid>
              {isEp && history.energies && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    Final System Energy
                  </Typography>
                  <Typography variant="h6" color="success">
                    {history.energies[history.energies.length - 1]?.toFixed(4)}
                  </Typography>
                </Grid>
              )}
            </Grid>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Training Interface
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Start training with Equilibrium Propagation and Backpropagation to compare their performance.
        Monitor training progress and view results in real-time.
      </Typography>

      <Grid container spacing={3}>
        {/* EP Training Form */}
        <Grid item xs={12} md={6}>
          {renderTrainingForm(
            'ep',
            epParams,
            setEpParams,
            startEpTraining,
            stopEpTraining,
            status.ep_training
          )}
          {renderResults('ep', epHistory)}
        </Grid>

        {/* Backprop Training Form */}
        <Grid item xs={12} md={6}>
          {renderTrainingForm(
            'backprop',
            backpropParams,
            setBackpropParams,
            startBackpropTraining,
            stopBackpropTraining,
            status.backprop_training
          )}
          {renderResults('backprop', backpropHistory)}
        </Grid>
      </Grid>

      {/* Status Overview */}
      <Paper elevation={2} sx={{ p: 3, mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Training Status Overview
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={status.ep_training ? <PlayArrow /> : <CheckCircle />}
                label={status.ep_training ? 'EP Training' : 'EP Ready'}
                color={status.ep_training ? 'warning' : 'success'}
              />
              {status.ep_results && (
                <Chip
                  icon={<CheckCircle />}
                  label="EP Complete"
                  color="success"
                  size="small"
                />
              )}
            </Box>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={status.backprop_training ? <PlayArrow /> : <CheckCircle />}
                label={status.backprop_training ? 'Backprop Training' : 'Backprop Ready'}
                color={status.backprop_training ? 'warning' : 'success'}
              />
              {status.backprop_results && (
                <Chip
                  icon={<CheckCircle />}
                  label="Backprop Complete"
                  color="success"
                  size="small"
                />
              )}
            </Box>
          </Grid>
        </Grid>

        {status.ep_results && status.backprop_results && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Both training runs are complete! Navigate to the Comparison page to analyze the results.
          </Alert>
        )}
      </Paper>
    </Container>
  );
};

export default Training;
