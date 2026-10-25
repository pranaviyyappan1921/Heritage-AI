# HeritageAI Launcher
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "        Starting HeritageAI (Ancient Tamil Manuscript)" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host ""
Set-Location "$PSScriptRoot"
python -m streamlit run app.py
