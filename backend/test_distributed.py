import json
import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.distributed import build_deepspeed_zero2_config, validate_distributed_strategy, validate_single_gpu_training


class DistributedConfigTests(unittest.TestCase):
    def test_ddp_is_valid_for_single_gpu(self) -> None:
        self.assertEqual(validate_distributed_strategy("ddp", 1), "ddp")

    def test_zero2_requires_multiple_gpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要 2 张 GPU"):
            validate_distributed_strategy("deepspeed_zero2", 1)

    def test_current_production_boundary_rejects_multi_gpu(self) -> None:
        validate_single_gpu_training(1, "single")
        with self.assertRaisesRegex(ValueError, "暂不支持多卡训练"):
            validate_single_gpu_training(2, "ddp")

    def test_resource_resolver_rejects_multi_gpu_before_provider_lookup(self) -> None:
        from backend import app as app_module

        payload = app_module.YouyunResourceSelection(
            gpuCount=2,
            distributedStrategy="ddp",
        )
        with patch.object(app_module, "build_youyun_runner") as build_runner:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(app_module.resolve_youyun_resource(payload))
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("暂不支持多卡训练", str(raised.exception.detail))
        build_runner.assert_not_called()

    def test_zero2_config_is_stage_two_and_uses_auto_batch_values(self) -> None:
        config = json.loads(build_deepspeed_zero2_config())
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertEqual(config["train_batch_size"], "auto")
        self.assertTrue(config["zero_optimization"]["reduce_scatter"])


if __name__ == "__main__":
    unittest.main()
