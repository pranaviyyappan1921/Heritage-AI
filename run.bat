@echo off
title HeritageAI - Ancient Tamil Manuscript Decoder
echo ========================================================
echo        Starting HeritageAI (Semmozhi Sudar)
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
