@echo off
echo ============================================
echo   Restarting LangFlow
echo ============================================
echo.

docker-compose down
docker-compose up -d --build

echo.
echo ============================================
echo   LangFlow restarted successfully.
echo.
echo   Open in your browser:
echo   http://localhost:1234
echo ============================================
echo.
echo Showing container logs (Ctrl+C to exit)...
echo.

docker-compose logs -f
