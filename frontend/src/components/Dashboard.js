import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  TrendingUp,
  Science,
  Compare,
  CheckCircle,
  Speed,
  Psychology,
} from '@mui/icons-material';

const Dashboard = () => {
  const features = [
    {
      icon: <Science color="primary" />,
      title: 'Equilibrium Propagation',
      description: 'Energy-based alternative to backpropagation using equilibrium states',
    },
    {
      icon: <Compare color="secondary" />,
      title: 'Side-by-Side Comparison',
      description: 'Compare EP performance against standard backpropagation',
    },
    {
      icon: <CheckCircle color="success" />,
      title: 'MNIST Validation',
      description: 'Validated on MNIST dataset with 3% gradient accuracy target',
    },
    {
      icon: <Speed color="info" />,
      title: 'Real-time Training',
      description: 'Monitor training progress and energy dynamics in real-time',
    },
    {
      icon: <Psychology color="warning" />,
      title: 'Biologically Plausible',
      description: 'Learning rules that mimic biological neural networks',
    },
    {
      icon: <TrendingUp color="error" />,
      title: 'Performance Metrics',
      description: 'Comprehensive analysis of convergence and energy dynamics',
    },
  ];

  const keyMetrics = [
    { label: 'Gradient Accuracy', value: '≤3%', color: 'success' },
    { label: 'Dataset', value: 'MNIST', color: 'primary' },
    { label: 'Architecture', value: 'MLP', color: 'secondary' },
    { label: 'Implementation', value: 'PyTorch', color: 'info' },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Hero Section */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          Equilibrium Propagation
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          An Alternative to Backpropagation Using Energy-Based Dynamics
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto' }}>
          This project implements and validates Equilibrium Propagation (EP), demonstrating that 
          EP can achieve gradient signals within 3% of backpropagation on the MNIST dataset. 
          The implementation provides reproducible tooling for side-by-side comparisons of EP's 
          convergence and energy-based dynamics against backpropagation.
        </Typography>
      </Box>

      {/* Key Metrics */}
      <Paper elevation={2} sx={{ p: 3, mb: 6 }}>
        <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
          Key Metrics
        </Typography>
        <Grid container spacing={2}>
          {keyMetrics.map((metric) => (
            <Grid item xs={6} sm={3} key={metric.label}>
              <Box sx={{ textAlign: 'center' }}>
                <Chip
                  label={metric.value}
                  color={metric.color}
                  variant="filled"
                  sx={{ fontSize: '1.2rem', py: 1, px: 2 }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {metric.label}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Features Grid */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Project Features
      </Typography>
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h6" component="h3" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* How It Works */}
      <Paper elevation={2} sx={{ p: 3, mb: 6 }}>
        <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
          How Equilibrium Propagation Works
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Traditional Backpropagation
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="primary" />
                </ListItemIcon>
                <ListItemText primary="Forward pass through network" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="primary" />
                </ListItemIcon>
                <ListItemText primary="Compute loss and gradients" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="primary" />
                </ListItemIcon>
                <ListItemText primary="Update weights using gradients" />
              </ListItem>
            </List>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Equilibrium Propagation
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <Science color="secondary" />
                </ListItemIcon>
                <ListItemText primary="Compute free phase equilibrium" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Science color="secondary" />
                </ListItemIcon>
                <ListItemText primary="Compute nudged phase equilibrium" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Science color="secondary" />
                </ListItemIcon>
                <ListItemText primary="Derive gradients from state differences" />
              </ListItem>
            </List>
          </Grid>
        </Grid>
      </Paper>

      {/* Benefits */}
      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
          Benefits of Equilibrium Propagation
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom color="primary">
              Biological Plausibility
            </Typography>
            <Typography variant="body2" color="text.secondary">
              EP uses local learning rules that are more biologically plausible than 
              backpropagation, making it a better model for understanding biological 
              neural networks.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom color="secondary">
              Energy-Based Learning
            </Typography>
            <Typography variant="body2" color="text.secondary">
              The algorithm minimizes an energy function, providing a principled 
              approach to learning that can be extended to various architectures 
              and learning scenarios.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom color="success">
              Comparable Performance
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Despite being biologically plausible, EP achieves performance comparable 
              to backpropagation, making it a viable alternative for practical applications.
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default Dashboard;
