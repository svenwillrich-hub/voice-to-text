@echo off
echo ============================================
echo   Starting Voice-to-Text
echo ============================================
echo.

docker-compose up -d --build

echo.
echo ============================================
echo   Voice-to-Text is starting...
echo.
echo   Open in your browser:
echo   http://localhost
echo.
echo   API:  http://localhost:8000/health
echo ============================================
echo.
echo Showing container logs (Ctrl+C to exit)...
echo.

docker-compose logs -f
