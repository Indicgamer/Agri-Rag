@echo off
cd /d "%~dp0"
python -m streamlit run ui\lightweight_demo.py --server.port 8507 --server.headless true --browser.gatherUsageStats false
