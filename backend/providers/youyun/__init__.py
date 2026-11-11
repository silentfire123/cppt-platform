"""优云（CompShare）远程 GPU Provider。

按需求第 10 节拆分：签名 signer、数据模型 models、API 客户端 client、
实例生命周期 instance_manager、对象存储 storage、SSH 通道 ssh。
凭据只允许来自 .env / Drone Secrets，禁止入库、禁止写日志。
"""
