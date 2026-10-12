"""
ForgeSight — Main Application Launcher
Starts the Uvicorn FastAPI server and launches the operations dashboard.
"""

import uvicorn
import webbrowser
import os
import sys

def main():
    print("=" * 65)
    print("  FORGESIGHT // Industrial AI Quality-Control Engine")
    print("  Computer Vision · Edge AI · MLOps · Production Server")
    print("=" * 65)
    print("  Web Dashboard: http://localhost:8000")
    print("  API Docs:      http://localhost:8000/docs")
    print("=" * 65)

    # Launch uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
