# TopologicPy Web Backend

## Local development (Windows)

Start the backend server with the local virtual environment:

```powershell
cd "C:/Users/lmurugesan/OneDrive - Alfaisal University/CM-iTAD/topologic_webapp/TopologicStudio/topologicpy-web-backend"
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If port 8000 is already in use, choose another port and update `API_BASE` in `topologicpy-web-frontend/src/App.jsx`.
