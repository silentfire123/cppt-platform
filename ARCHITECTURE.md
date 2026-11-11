# 架构与关键链路

## 1. 系统定位

平台是模型训练与推理任务的控制面：前端负责输入与状态展示，FastAPI 校验业务请求并编排执行，云端 GPU 或可选本地环境承载实际计算。API 服务健康不等于 GPU、模型或训练环境就绪。

## 2. 模块职责

| 层次 | 入口 | 职责 |
| --- | --- | --- |
| 前端 | `app/page.tsx`、`components/` | 模块导航、表单、任务详情、日志及推理交互 |
| 前端契约 | `lib/types.ts`、`lib/api-config.ts` | 数据类型、API 地址和状态处理 |
| API | `backend/app.py` | 数据集/任务接口、参数检查、队列处理、运行生命周期 |
| 训练配置 | `backend/llamafactory.py`、`backend/local_llamafactory_executor.py` | 参数转换、训练配置、执行与产物发现 |
| 云端接入 | `backend/providers/youyun/` | Provider 签名、资源发现、实例生命周期、SSH 与远程执行 |
| 任务存储 | `backend/store.py`、`backend/postgres_storage.py` | 任务、日志、指标与模型目录持久化 |
| 模型产物 | `backend/training_artifacts.py`、`backend/model_storage.py` | Best / Final 发现、上传与恢复、文件完整性验证 |
| 推理 | `backend/inference_service.py`、`backend/providers/youyun/inference.py` | 服务管理、启动参数与日志、远程推理 |
| 横切能力 | `backend/auth.py`、`backend/governance.py`、`backend/retention_cleanup.py` | 认证、审计、运行记录与保留清理 |

## 3. 训练链路

1. 用户上传 JSONL 数据，完成结构校验并发布；任务类型需要与 SFT / DPO 数据结构匹配。
2. API 校验模型来源、数据集和训练参数，保存任务，进入队列。
3. worker 选择执行路径，转换 LLaMA-Factory 配置；远程路径负责实例和 SSH 准备。
4. 训练日志和指标回传并持久化，前端经 HTTP / WebSocket 展示任务进度。
5. 训练产物被识别为 Best / Final 等资产，运行路径与备份状态单独记录。
6. 配置了对象存储后执行上传；实例保留期与释放逻辑独立处理，避免把“训练完成”混同为“备份完成”。

## 4. 推理链路

从模型资产确定基座与 Adapter，优先使用完整的可访问文件。文件缺失且具有备份清单时，经对象存储恢复并验证，再由推理运行时启动服务。服务生命周期和请求级生成参数分离；流式接口与日志通道各自传递结果和运行信息。

## 5. 存储与执行边界

- SQLite 用于便于启动的本地模式；PostgreSQL 用于共享存储。模型大文件不应存入关系数据库。
- 本地运行路径、云实例上的路径和 MinIO 对象前缀具有不同含义，不能相互替代。
- GPU 计算、API 调度和对象存储是独立的故障域；接口返回、作业退出、产物完整与远端释放需要分别判断。
- 多 worker / 多副本控制面、跨实例调度公平性和完整多机训练不属于当前实现范围。

## 6. 环境与数据配置

环境配置通过 `.env` 注入，示例文件使用本地或占位地址。Docker 后端负责 API 与任务编排，计算任务在单独的运行环境中执行。仓库内的数据为合成样例，数据集目录及 ID 沿用现有接口约定。

关于启动、配置和测试，以根目录 [README](README.md) 为准。
