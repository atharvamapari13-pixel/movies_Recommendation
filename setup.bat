@echo off

echo ====================================
echo Creating Virtual Environment...
echo ====================================
python -m venv venv

echo.
echo ====================================
echo Activating Virtual Environment...
echo ====================================
call venv\Scripts\activate

echo.
echo ====================================
echo Upgrading pip...
echo ====================================
python -m pip install --upgrade pip

echo.
echo ====================================
echo Installing Required Packages...
echo ====================================
pip install -r requirements.txt

echo.
echo ====================================
echo Installation Complete!
echo ====================================

echo.
echo To run backend:
echo uvicorn main:app --reload

echo.
echo To run frontend:
echo streamlit run app.py

echo.
pause