@echo off
echo Starting ForgeSight Industrial AI Platform...
python -m backend.generate_samples
python run_server.py
pause
