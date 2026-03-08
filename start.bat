@echo off
echo ============================================
echo   Starting LangFlow
echo ============================================
echo.

docker-compose up -d --build

echo.
echo ============================================
echo   LangFlow is starting...
echo.
echo   Open in your browser:
echo   http://localhost:1234
echo.
echo   API:  http://localhost:8000/health
echo ============================================
echo.
echo Showing container logs (Ctrl+C to exit)...
echo.

docker-compose logs -f
