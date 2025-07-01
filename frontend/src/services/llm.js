import axios from 'axios';

const openaiBase = 'https://api.openai.com/v1/chat/completions';

/**
 * Call a hosted LLM (default: OpenAI GPT-3.5 / GPT-4) directly from the browser.
 * The API key should be supplied via Vite env var (VITE_OPENAI_API_KEY).
 * Model can be overridden with VITE_OPENAI_MODEL.
 *
 * @param {string} prompt
 * @returns {Promise<string>} assistant response
 */
export async function callLLM(prompt) {
  const apiKey = '';
  const model = 'gpt-3.5-turbo';

  if (!apiKey) {
    // Fallback for development without a key – simply echo the prompt.
    return Promise.resolve('🔧 [LLM stub] Echo: ' + prompt);
  }

  try {
    const res = await axios.post(
      openaiBase,
      {
        model,
        messages: [
          { role: 'system', content: 'You are a helpful financial assistant.' },
          { role: 'user', content: prompt },
        ],
      },
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
      }
    );
    return res.data.choices?.[0]?.message?.content?.trim() || '';
  } catch (err) {
    console.error('LLM error', err);
    const detail = err.response?.data?.error?.message || err.message;
    return `⚠️ ${detail}`;
  }
}
