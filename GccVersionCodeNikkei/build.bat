@echo off
rem 画面をクリア
cls

rem コンパイル実行（*.c でフォルダ内の全Cファイルを対象にする）
echo Compiling...
gcc *.c -o neuralNetNikkei.exe

rem エラーがあったら止まる
if %errorlevel% neq 0 (
    echo [Error] Compile failed.
    pause
    exit /b
)

rem 成功したら実行する
echo [Success] Running neuralNetNikkei...
echo ------------------------------------------
neuralNetNikkei.exe
echo.
echo ------------------------------------------
pause