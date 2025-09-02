import React from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Link,
} from '@mui/material';
import {
  Science,
  Psychology,
  TrendingUp,
  Code,
  School,
  GitHub,
  Article,
} from '@mui/icons-material';

const About = () => {
  const technicalDetails = [
    {
      title: 'Algorithm Implementation',
      items: [
        'Energy-based neural network architecture',
        'Equilibrium state computation through iterative updates',
        'Gradient computation via state differences',
        'Free phase and nudged phase equilibrium',
      ],
    },
    {
      title: 'Network Architecture',
      items: [
        'Multi-layer perceptron (MLP)',
        'Configurable hidden layer sizes',
        'ReLU activation functions',
        'Cross-entropy loss for classification',
      ],
    },
    {
      title: 'Training Process',
      items: [
        'MNIST dataset (70k handwritten digits)',
        'Configurable learning rates and epochs',
        'Real-time progress monitoring',
        'Automatic model checkpointing',
      ],
    },
  ];

  const advantages = [
    {
      title: 'Biological Plausibility',
      description: 'EP uses local learning rules that are more biologically plausible than backpropagation, making it a better model for understanding biological neural networks.',
      icon: <Psychology color="primary" />,
    },
    {
      title: 'Energy-Based Learning',
      description: 'The algorithm minimizes an energy function, providing a principled approach to learning that can be extended to various architectures and learning scenarios.',
      icon: <Science color="secondary" />,
    },
    {
      title: 'Comparable Performance',
      description: 'Despite being biologically plausible, EP achieves performance comparable to backpropagation, making it a viable alternative for practical applications.',
      icon: <TrendingUp color="success" />,
    },
    {
      title: 'Reproducible Research',
      description: 'This implementation provides reproducible tooling for comparing EP against backpropagation, enabling researchers to validate and extend the work.',
      icon: <Code color="info" />,
    },
  ];

  const researchContext = [
    'Alternative to backpropagation for neural network training',
    'Energy-based learning algorithms',
    'Biologically plausible neural network models',
    'Local learning rules in neural networks',
    'Equilibrium dynamics in neural systems',
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          About Equilibrium Propagation
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Understanding the Theory, Implementation, and Applications
        </Typography>
      </Box>

      {/* Theory Section */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Theoretical Foundation
        </Typography>
        <Typography variant="body1" paragraph>
          Equilibrium Propagation (EP) is an alternative to backpropagation that uses energy-based dynamics 
          instead of gradient descent. The algorithm works by computing equilibrium states of a neural network 
          and deriving gradients from the differences between these states.
        </Typography>
        
        <Grid container spacing={3} sx={{ mt: 3 }}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Core Principles
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <Science color="primary" />
                </ListItemIcon>
                <ListItemText primary="Energy minimization through equilibrium states" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Science color="primary" />
                </ListItemIcon>
                <ListItemText primary="Local learning rules for biological plausibility" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Science color="primary" />
                </ListItemIcon>
                <ListItemText primary="Gradient computation via state perturbations" />
              </ListItem>
            </List>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Mathematical Framework
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              EP minimizes an energy function E(x, y, θ) where x is input, y is output, and θ are parameters. 
              The algorithm computes two equilibrium states: free phase (no target) and nudged phase (with target), 
              then derives gradients as:
            </Typography>
            <Box sx={{ 
              backgroundColor: 'grey.100', 
              p: 2, 
              borderRadius: 1, 
              fontFamily: 'monospace',
              textAlign: 'center'
            }}>
              ∇θ = (y*_nudged - y*_free) / β
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Implementation Details */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Implementation Details
        </Typography>
        <Typography variant="body1" paragraph>
          This project provides a complete PyTorch implementation of Equilibrium Propagation, including 
          the neural network architecture, training loops, and comparison tools against backpropagation.
        </Typography>
        
        <Grid container spacing={3}>
          {technicalDetails.map((section, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {section.title}
                  </Typography>
                  <List dense>
                    {section.items.map((item, itemIndex) => (
                      <ListItem key={itemIndex} sx={{ py: 0.5 }}>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Advantages */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Advantages Over Backpropagation
        </Typography>
        <Grid container spacing={3}>
          {advantages.map((advantage, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {advantage.icon}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      {advantage.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {advantage.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Research Context */}
      <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Research Context
        </Typography>
        <Typography variant="body1" paragraph>
          Equilibrium Propagation addresses several important challenges in deep learning and computational neuroscience:
        </Typography>
        
        <Grid container spacing={2}>
          {researchContext.map((context, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Chip
                label={context}
                variant="outlined"
                color="primary"
                sx={{ m: 0.5 }}
              />
            </Grid>
          ))}
        </Grid>
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h6" gutterBottom>
          Current Research Areas
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          EP is actively researched in areas such as neuromorphic computing, brain-inspired AI, 
          and alternative training algorithms for neural networks. This implementation contributes 
          to the reproducibility and validation of EP research.
        </Typography>
      </Paper>

      {/* Project Information */}
      <Paper elevation={2} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>
          Project Information
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              <School sx={{ mr: 1, verticalAlign: 'middle' }} />
              Academic Context
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              This project was developed as part of research into alternative neural network training algorithms. 
              It demonstrates the implementation and validation of Equilibrium Propagation on the MNIST dataset, 
              achieving the target of gradient signals within 3% of backpropagation.
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              <Code sx={{ mr: 1, verticalAlign: 'middle' }} />
              Technical Implementation
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Built with PyTorch for the backend algorithms and React for the frontend interface. 
              The system provides real-time training monitoring, comprehensive result analysis, 
              and side-by-side comparisons between EP and backpropagation.
            </Typography>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 3 }} />
        
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            Get Involved
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Chip
              icon={<GitHub />}
              label="View on GitHub"
              component={Link}
              href="https://github.com/SagnikBiswasDL/equilibrium-propagation"
              target="_blank"
              clickable
              color="primary"
              variant="outlined"
            />
            <Chip
              icon={<Article />}
              label="Read the Paper"
              component={Link}
              href="#"
              clickable
              color="secondary"
              variant="outlined"
            />
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default About;
