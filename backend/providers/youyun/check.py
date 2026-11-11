"""优云凭据联通性检查：python -m backend.providers.youyun.check

分步引导，输出不打印任何密钥：
1. 只有公钥/私钥 → 验证签名可用，自动列出支持的可用区，给出待填配置行。
2. 补齐区域/可用区 → 列出 GPU 机型（含售罄状态）与可用镜像，给出镜像配置行。
3. 配置齐全 → 完整库存检查。

退出码：0 全部通过；2 凭据有效但配置不全（按提示补填后重跑）；1 失败。
"""
from __future__ import annotations

import sys

from .client import YouyunClient
from .models import YouyunApiError, YouyunConfig

_IMAGE_KEYWORDS = ("llama-factory", "llamafactory", "cuda", "pytorch", "vllm", "torch")


def _suggest(lines: list[str]) -> int:
    print("\n请把以下配置行填入项目根目录 .env（等号右边改成你要的值），然后重跑本脚本：")
    for line in lines:
        print(f"  {line}")
    return 2


def main() -> int:
    config = YouyunConfig.from_env()
    if not config.public_key or not config.private_key:
        print("[待填] .env 中还缺公钥/私钥，请填这两行（值向管理员/老师索取）：")
        print("  YOUYUN_PUBLIC_KEY=把老师给的公钥粘贴在这里")
        print("  YOUYUN_PRIVATE_KEY=把老师给的私钥粘贴在这里")
        return 2
    print(f"[OK] 公钥/私钥已填。base_url={config.api_base_url}")

    try:
        client = YouyunClient(config, require_full=False)
        zones = client.describe_support_zones()
    except YouyunApiError as exc:
        print(f"[FAIL] {exc}")
        print("      若提示签名/认证错误，请核对公钥私钥是否复制完整（无多余空格换行）。")
        return 1

    china_zones = [z for z in zones if z["region"].startswith("cn-")]
    candidates = china_zones or zones
    print(f"[OK] 凭据有效，签名验证通过。账号支持 {len(zones)} 个可用区：")
    for z in zones:
        pod = "（Pod 型，不适用于整机训练）" if z["is_pod"] else ""
        print(f"     {z['zone']:<14} {z['describe']} {pod}")

    missing = config.missing_keys()
    if "YOUYUN_REGION" in missing or "YOUYUN_ZONE" in missing:
        preferred = next((z for z in candidates if not z["is_pod"]), candidates[0] if candidates else None)
        if preferred:
            return _suggest(
                [
                    f"YOUYUN_REGION={preferred['region']}",
                    f"YOUYUN_ZONE={preferred['zone']}",
                ]
            )
        return 1

    print(f"[OK] 区域/可用区已配置：{config.region} / {config.zone}")

    try:
        types = client.describe_instance_types(config.gpu_type)
    except YouyunApiError as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not types:
        print(f"[WARN] 区域 {config.region} 未返回 {config.gpu_type} 机型，可能不支持，请换 GPU_TYPE 或区域。")
        return 1
    for item in types:
        state = "可售" if not item.sold_out else "售罄"
        print(f"[OK] 机型 {item.name}（{item.zone}）：{state} 显存={item.graphics_memory_gb}GB {item.description}")

    if not config.training_image_id and not config.image_id:
        try:
            images = client.describe_images()
        except YouyunApiError as exc:
            print(f"[FAIL] 镜像列表查询失败：{exc}")
            return 1
        matched = [
            img
            for img in images
            if any(kw in (img["name"] + " " + " ".join(img["tags"]) + " " + img["description"]).lower() for kw in _IMAGE_KEYWORDS)
        ] or images

        def image_priority(img: dict) -> tuple[int, str]:
            text = img["name"].lower()
            if "torch" in text:
                return 0, text
            if "cuda" in text:
                return 1, text
            if "nvidia" in text:
                return 2, text
            return 3, text

        matched = sorted(matched, key=image_priority)
        print(f"[OK] 平台镜像共 {len(images)} 个，与训练相关的候选（LLaMA-Factory / CUDA / PyTorch / vLLM）：")
        for img in matched[:15]:
            tags = ",".join(img["tags"]) or "-"
            print(f"     {img['image_id']:<28} {img['name']}  [{tags}] {img['status']}")
        if matched:
            return _suggest([f"YOUYUN_TRAINING_IMAGE_ID={matched[0]['image_id']}   # 或从上面候选里换一个"])
        return 1

    try:
        capacity = client.check_resource_capacity(image_id=config.training_image_id or config.image_id)
    except YouyunApiError as exc:
        print(f"[FAIL] 库存检查失败：{exc}")
        return 1
    available = capacity.first_available()
    for spec in capacity.specs:
        print(
            f"     规格 {spec.gpu}卡/{spec.cpu}核/{spec.memory_gb}GB："
            + ("有库存" if spec.resource_enough else "无库存")
        )
    if available:
        print(f"[OK] 库存检查通过，推荐规格：{available.gpu}卡/{available.cpu}核/{available.memory_gb}GB")
        print("[OK] 优云 Provider 配置完成，可以创建实例。")
        return 0
    print("[FAIL] 当前规格全部无库存，可换 GPU_TYPE / 区域 / 计费模式重试。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
