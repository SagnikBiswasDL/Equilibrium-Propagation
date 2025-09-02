import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Alert,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
} from '@mui/material';
import {
  Compare,
  TrendingUp,
  Science,
  CheckCircle,
  Warning,
  Refresh,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import axios from 'axios';

const Comparison = () => {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/compare');
      setComparisonData(response.data.comparison);
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to generate comparison');
    } finally {
      setLoading(false);
    }
  };

  const fetchExistingComparison = async () => {
    try {
      const response = await axios.get('/api/results/comparison');
      setComparisonData(response.data);
    } catch (error) {
      // No existing comparison data
      setComparisonData(null);
    }
  };

  useEffect(() => {
    fetchExistingComparison();
  }, []);

  const renderGradientComparison = () => {
    if (!comparisonData?.gradient_comparison) return null;

    const grad = comparisonData.gradient_comparison;
    const isTargetMet = grad.mean_difference_percent <= 3.0;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <TrendingUp color="primary" />
            <Typography variant="h6" sx={{ ml: 1 }}>
              Gradient Signal Comparison
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Target Achievement
              </Typography>
              <Chip
                icon={isTargetMet ? <CheckCircle /> : <Warning />}
                label={isTargetMet ? 'Target Met!' : 'Target Not Met'}
                color={isTargetMet ? 'success' : 'warning'}
                sx={{ fontSize: '1.1rem', py: 1, px: 2 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                EP gradients within {grad.mean_difference_percent?.toFixed(2)}% of backprop
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Key Metrics
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Mean Difference</TableCell>
                      <TableCell>{grad.mean_difference_percent?.toFixed(2)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Max Difference</TableCell>
                      <TableCell>{grad.max_difference_percent?.toFixed(2)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Correlation</TableCell>
                      <TableCell>{grad.gradient_correlation?.toFixed(3)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderConvergenceComparison = () => {
    if (!comparisonData?.convergence_comparison) return null;

    const conv = comparisonData.convergence_comparison;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <TrendingUp color="secondary" />
            <Typography variant="h6" sx={{ ml: 1 }}>
              Convergence Comparison
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Final Accuracies
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>EP Training</TableCell>
                      <TableCell>{conv.ep_final_train_acc?.toFixed(2)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>EP Validation</TableCell>
                      <TableCell>{conv.ep_final_val_acc?.toFixed(2)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Backprop Training</TableCell>
                      <TableCell>{conv.bp_final_train_acc?.toFixed(2)}%</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Backprop Validation</TableCell>
                      <TableCell>{conv.bp_final_val_acc?.toFixed(2)}%</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Performance Analysis
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Training Difference</TableCell>
                      <TableCell>
                        {conv.train_acc_difference > 0 ? '+' : ''}
                        {conv.train_acc_difference?.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Validation Difference</TableCell>
                      <TableCell>
                        {conv.val_acc_difference > 0 ? '+' : ''}
                        {conv.val_acc_difference?.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Convergence Similarity</TableCell>
                      <TableCell>{conv.convergence_similarity?.toFixed(3)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderEnergyAnalysis = () => {
    if (!comparisonData?.energy_analysis) return null;

    const energy = comparisonData.energy_analysis;

    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Science color="success" />
            <Typography variant="h6" sx={{ ml: 1 }}>
              Energy Dynamics Analysis
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Energy Statistics
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>Final Energy</TableCell>
                      <TableCell>{energy.final_energy?.toFixed(4)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Decay Rate</TableCell>
                      <TableCell>{energy.energy_decay_rate?.toFixed(6)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Stability (Std Dev)</TableCell>
                      <TableCell>{energy.energy_stability?.toFixed(4)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Energy Evolution
              </Typography>
              {energy.energies && energy.energies.length > 0 && (
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={energy.energies.map((e, i) => ({ epoch: i + 1, energy: e }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="energy" stroke="#4caf50" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  };

  const renderSummary = () => {
    if (!comparisonData?.summary) return null;

    return (
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Comparison Summary
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(comparisonData.summary).map(([key, value]) => (
            <Grid item xs={12} key={key}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  icon={value.includes('✅') ? <CheckCircle /> : <Warning />}
                  label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  color={value.includes('✅') ? 'success' : 'warning'}
                  variant="outlined"
                />
                <Typography variant="body1" sx={{ ml: 1 }}>
                  {value}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Paper>
    );
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center' }}>
          <LinearProgress sx={{ mb: 2 }} />
          <Typography>Generating comparison...</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" component="h1">
          Results Comparison
        </Typography>
        <Button
          variant="contained"
          startIcon={<Refresh />}
          onClick={fetchComparison}
          disabled={loading}
        >
          Generate Comparison
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {!comparisonData ? (
        <Alert severity="info" sx={{ mb: 3 }}>
          No comparison data available. Complete both EP and backprop training first, then generate a comparison.
        </Alert>
      ) : (
        <>
          {renderSummary()}
          {renderGradientComparison()}
          {renderConvergenceComparison()}
          {renderEnergyAnalysis()}
        </>
      )}

      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          About the Comparison
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          This comparison analyzes the performance of Equilibrium Propagation against standard backpropagation.
          The key metric is whether EP can achieve gradient signals within 3% of backpropagation, which would
          validate EP as a viable alternative learning algorithm.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          The analysis includes gradient accuracy, convergence rates, and energy dynamics specific to EP.
          Use this data to understand the strengths and limitations of each approach.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Comparison;
