import { useState } from 'react';
import { callLLM } from '../services/llm';
import { Drawer, Box, Typography, TextField, Button } from '@mui/material';

const ChatPrompt = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');


  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    const assistantContent = await callLLM(input);
    const assistantMsg = { role: 'assistant', content: assistantContent };
    setMessages(prev => [...prev, assistantMsg]);
  };

  // No close handler needed – chat stays open persistently

  return (
    <Drawer
      anchor="right"
      variant="persistent"
      open
      sx={{
        width: 450,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          top: '64px',
          height: 'calc(100% - 64px)',
          width: 450,
          boxSizing: 'border-box',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          borderRadius: '10px 0 0 10px',
          background: '#242424',
          boxShadow: '0 0 12px rgba(100,108,255,0.6)',
        },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="h4" sx={{mt: 2, fontWeight: 'bold', textAlign: 'center' }}>
          Ask Anything
        </Typography>
      </Box>
      <Box sx={{ flex: 1, overflowY: 'auto', display:'flex', flexDirection:'column', justifyContent:'flex-end' }}>
        {messages.map((m, idx) => (
          <Box key={idx} sx={{
            mb: 1,
            textAlign: m.role === 'user' ? 'right' : 'left',
          }}>
            <Typography
              sx={{
                display: 'inline-block',
                p: 1.2,
                px: 1.8,
                borderRadius: '12px',
                background: m.role === 'user' ? '#646cff' : '#333',
                color: m.role === 'user' ? '#fff' : '#e0e0e0',
                boxShadow: '0 0 6px rgba(100,108,255,0.4)',
                maxWidth: '90%',
                whiteSpace: 'pre-wrap',
              }}
            >
              {m.content}
            </Typography>
          </Box>
        ))}
      </Box>
      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          multiline
          minRows={2}
          fullWidth
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question..."
        />
        <Button variant="contained" onClick={handleSend}>Send</Button>
      </Box>
    </Drawer>
  );
};

export default ChatPrompt;

