@echo off
title EDN1 v2 — Assistant IA Saisines
color 0A

echo.
echo  ================================================
echo   EDN1 v2 — Assistant IA Saisines
echo  ================================================
echo.

:: Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERREUR] Python n'est pas installe.
    echo  Telechargez-le sur https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Installer les dependances si necessaire
echo  [1/3] Verification des dependances...
pip install -r requirements.txt -q
echo  OK

:: Verifier que le fichier Parquet existe
if not exist "data\processed\saisines.parquet" (
    echo.
    echo  [2/3] Premiere utilisation : preparation des donnees...
    echo        Cela prend environ 2 minutes.
    echo.
    python pipeline\01_extract.py
    python pipeline\02_classify.py
    python pipeline\03_to_parquet.py
    echo  OK - Donnees prets
) else (
    echo  [2/3] Donnees deja preparees. OK
)

:: Verifier Ollama
echo  [3/3] Verification d'Ollama...
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ATTENTION] Ollama ne semble pas etre lance.
    echo  Lancez Ollama depuis la barre des taches, puis appuyez sur une touche.
    pause
)

:: Lancer le dashboard
echo.
echo  Lancement du dashboard...
echo  Le navigateur va s'ouvrir automatiquement.
echo  Pour arreter : fermez cette fenetre ou appuyez sur Ctrl+C
echo.
start "" http://localhost:8501
streamlit run app/Home.py --server.headless false

pause
