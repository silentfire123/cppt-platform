"""优云 API 公钥/私钥签名（与官方 ucloud-sdk-python3 的 verify_ac 一致）。

规则：参数拍平后按 key 排序，把 str(key)+encode_value(value) 逐项拼接，
末尾接私钥，整体 UTF-8 后取 SHA-1 十六进制。无 URL 编码、无分隔符。
"""
from __future__ import annotations

import hashlib


def encode_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value % 1 == 0:
        return str(int(value))
    return str(value)


def signature_base(params: dict[str, object], private_key: str) -> str:
    return "".join(str(key) + encode_value(params[key]) for key in sorted(params)) + private_key


def sign(params: dict[str, object], private_key: str) -> dict[str, str]:
    """返回带 Signature 的扁平参数字典，供 POST 表单使用。"""
    digest = hashlib.sha1(signature_base({k: v for k, v in params.items() if k != "Signature"}, private_key).encode("utf-8")).hexdigest()
    return {**{str(k): encode_value(v) for k, v in params.items()}, "Signature": digest}
