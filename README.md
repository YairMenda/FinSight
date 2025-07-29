# FinSight

FinSight is a full-stack FinTech web application that lets users search for stocks, view real-time quotes and historical charts, and see a simple next-day price prediction.

## Project Structure
```
fin-sight/
├── frontend/          # React + Vite client
│   └── …
├── backend/           # Express REST API
│   └── …
└── README.md
```

## Tech Stack
* **Frontend:** React 19, React-Router, Axios, Recharts, MUI
* **Backend:** Node.js 20, Express, Yahoo-Finance2, Day.js

## Setup
### Prerequisites
* Node.js ≥ 18

### 1. Clone & install
```bash
# install backend deps
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install flask yfinance pandas scikit-learn ta lightgbm

# install frontend deps
cd ../frontend
npm install
```

### 2. Environment variables
Copy `.env.example` inside `backend` to `.env` and adjust if necessary. Default port is `5000`.

```
PORT=5000
```

### 3. Run in development
Start both servers in two terminals:
```bash
# terminal 1 – backend
cd backend
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
python app.py

# terminal 2 – frontend
cd frontend
npm run dev
```
The React dev server proxies API calls (`/api/*`) to the backend automatically.

Open `http://localhost:5173` (default Vite port).

## Scripts
### Backend
* `python app.py` – start normally

### Frontend
* `npm run dev` – Vite dev server
* `npm run build` – production build
* `npm run preview` – preview the build

## Future Improvements
* 🔒 Authentication & favourites saved per user (PostgreSQL)
* 🌙 Dark / Light mode toggle
* 📈 More advanced ML models (e.g. XGBoost served from Python)
* ✅ Unit & integration tests (Vitest, Jest, Supertest, React Testing Library)
