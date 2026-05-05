@echo off
echo ==========================================
echo 1. Compiling C++ Code (Dynamic Link)
echo ==========================================
g++ -O2 -s -o test_code.exe %1

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed!
    exit /b %ERRORLEVEL%
)
echo Compilation Successful.
echo.

echo ==========================================
echo 2. Running Stage 1: Binary to JSON
echo ==========================================
python "python\extract_corrected.py" --input test_code.exe --output test_code.json

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Stage 1 failed! 
    exit /b %ERRORLEVEL%
)
echo.

echo ==========================================
echo 3. Running Stage 2: JSON to Model Features
echo ==========================================
python "python\extract_features.py" test_code.json

echo.
echo ==========================================
echo Workflow Complete!
echo ==========================================