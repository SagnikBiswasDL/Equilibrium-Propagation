import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import ScienceIcon from '@mui/icons-material/Science';

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/' },
    { label: 'Training', path: '/training' },
    { label: 'Comparison', path: '/comparison' },
    { label: 'About', path: '/about' },
  ];

  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        <ScienceIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 0, mr: 4 }}>
          Equilibrium Propagation
        </Typography>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              onClick={() => navigate(item.path)}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
        
        <Typography variant="body2" sx={{ opacity: 0.8 }}>
          Alternative to Backpropagation
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
