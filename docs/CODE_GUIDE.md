# 开发指南

## 设计背景

**问题**：直接使用训练脚本时，数据、参数、GPU 实例、日志与模型文件分散，任务失败后也难追踪。本项目把这些环节整合为有状态、可观察的操作流程。

**方案**：Next.js 展示任务与资源；FastAPI 负责校验、队列、存储与执行编排；云端 GPU 运行 LLaMA-Factory / vLLM；PostgreSQL 保存业务记录，MinIO 保存模型产物。SQLite 保留为本地启动方案。

**关键关注点**：任务状态、远程异常、日志连续性与产物可靠性共同决定了训练与推理工作流的稳定性。

## 从哪里读代码

| 话题 | 代码入口 | 关注的问题 |
| --- | --- | --- |
| 从提交到执行 | `backend/app.py`、`backend/training_executor.py` | 校验、入队和真实执行为什么分开？ |
| 参数如何生效 | `backend/llamafactory.py` | UI 字段如何转换为框架配置？ |
| 远程运行 | `backend/providers/youyun/remote_training.py`、`backend/providers/youyun/ssh.py` | 实例、SSH、依赖和训练进程分别怎么判断状态？ |
| 流式日志 | `backend/providers/youyun/log_follow.py`、`backend/test_youyun_training_log_stream.py` | 网络中断与进程退出有什么区别？ |
| 产物管理 | `backend/training_artifacts.py`、`backend/model_storage.py` | Best / Final 和本地/远端/备份状态为什么要分离？ |
| 存储选择 | `backend/store.py`、`backend/postgres_storage.py` | SQLite 与 PostgreSQL 分别解决什么问题？ |
| 前端可靠性 | `lib/training-submit-state.ts`、`components/training-module.tsx` | 失败、重试、轮询和模块切换如何呈现？ |

## 关键设计

### 1. 任务不是一个同步 HTTP 请求

API 的请求生命周期短于 GPU 作业。任务状态需要持久化，运行日志需要可恢复地读取，页面刷新不应成为作业的生命周期边界。

### 2. 文件路径不是模型可用性的充分条件

训练进程退出、目录存在、Adapter 文件完整、备份完成和推理环境可读取是不同条件。模型恢复与产物回归测试分别覆盖这些判断条件。

### 3. 远程异常需要分层定位

Provider API、实例就绪、SSH 连接、运行依赖、模型下载和训练进程都可能失败。`backend/test_youyun_remote_training.py` 与 `backend/test_youyun_inference.py` 覆盖相应的错误处理；不可恢复的错误需要保留上下文并交由用户处理。

### 4. 测试围绕边界而不仅是正常路径

相关覆盖见 `backend/test_training_artifact_regressions.py`、`backend/test_training_worker_concurrency.py`、`backend/test_inference_stream.py` 和 `lib/` 下的状态测试，重点检查异常路径、状态变化和恢复行为。

## 当前能力边界

- 支持单 GPU 文本生成训练和推理编排；完整多机分布式训练不属于当前实现范围。
- 评测相关功能默认关闭，完整评测闭环尚未完成。
- 本地无 GPU 模式用于浏览与接口演示，不能证明真实训练或推理成功。
- 仓库不附性能、吞吐或成本收益数据；只有实际测量和可核验记录才能作为数字依据。
- 合成样例只验证格式与流程，不是模型效果证明。

## 运行检查

1. 按 README 在干净环境中完成启动，确认前端、健康检查和接口文档。
2. 沿任务数据流核对对应的函数、存储记录与异常分支。
3. 检查本地测试输出，分别执行自动化回归与所需的云端 GPU 手动验收。
4. 执行云端任务前确认费用、实例、模型下载和凭证，结束后检查资源释放。
5. 不要把关闭认证的本地服务直接部署到公网。
