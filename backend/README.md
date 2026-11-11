# Backend

FastAPI API、任务编排、存储及远程 GPU 集成。

- [启动与测试](../README.md)
- [架构边界](../ARCHITECTURE.md)
- [项目讲解](../docs/INTERVIEW_GUIDE.md)

从仓库根目录执行 `python scripts/run_backend_linux.py`，使用 `.env` 提供配置。该入口也可运行于原生 Windows Python；`scripts/run_backend.py` 则是 Windows + WSL 专用入口。

API：`http://127.0.0.1:8010`，OpenAPI：`http://127.0.0.1:8010/docs`。
