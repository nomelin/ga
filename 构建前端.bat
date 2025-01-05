@echo off
rem 切换到包含Vue工程的fe目录
cd fe
rem 执行构建命令，生成生产环境文件（会输出到dist目录）
npm run build
echo Vue工程构建完成，生产环境文件已生成到fe/dist目录下。
pause