@echo off
echo Starting PCOS Detection System...
echo.

REM Try to activate virtual environment
call pcos_env\Scripts\activate.bat

REM Try different Python commands
echo Trying to run with python...
python --version
if %errorlevel% equ 0 (
    echo Python found! Installing dependencies...
    python -m pip install streamlit pandas numpy scikit-learn plotly
    echo Starting Streamlit app...
    python -m streamlit run app\streamlit_enhanced.py
    goto :end
)

echo Trying to run with py...
py --version
if %errorlevel% equ 0 (
    echo Python found! Installing dependencies...
    py -m pip install streamlit pandas numpy scikit-learn plotly
    echo Starting Streamlit app...
    py -m streamlit run app\streamlit_enhanced.py
    goto :end
)

echo Python not found. Please install Python first.
echo You can download Python from: https://www.python.org/downloads/

:end
pause

