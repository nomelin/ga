@echo off
rem 切换到fe/dist
cd fe/dist
rem 启动python http.server，端口8081
python -m http.server 8081
echo Python服务器已启动，可在浏览器中访问 http://localhost:8081 查看页面。
pause