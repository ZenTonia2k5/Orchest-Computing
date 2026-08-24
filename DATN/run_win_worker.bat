@echo off
echo ========================================================================
echo   ⚡ ORCHEST COMPUTING: WINDOWS EDGE WORKER NODE
echo ========================================================================
echo Starting Edge Worker Node... (broadcasting UDP beacon on port 59001)
echo.

python apps\worker_node.py

pause
