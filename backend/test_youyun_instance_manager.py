"""YouyunInstanceManager 生命周期与清理保证的单元测试（伪造客户端，不发真实请求）。"""
from __future__ import annotations

import unittest

from backend.providers.youyun.client import YouyunClient
from backend.providers.youyun.instance_manager import YouyunInstanceManager, _decode_password
from backend.providers.youyun.models import GpuSpecCombination, YouyunConfig, YouyunInstance


def _make_instance(state: str = "Running", ip: str | None = "1.2.3.4") -> YouyunInstance:
    return YouyunInstance(
        uhost_id="uhost-1",
        name="cppt-t1",
        state=state,
        region="cn-wlcb",
        zone="cn-wlcb-01",
        gpu=1,
        gpu_type="4090",
        cpu=16,
        memory_mb=65536,
        ip=ip,
        ssh_port=60022,
        ssh_user="root",
        password_b64="cGFzc3dvcmQ=",
        image_id="img",
        os_type="Linux",
        charge_type="Postpay",
        instance_type="UHost",
        raw={},
    )


class FakeClient(YouyunClient):
    def __init__(self, states: list[str] | None = None, specs: list[GpuSpecCombination] | None = None) -> None:  # noqa: super 不校验
        self.config = YouyunConfig(public_key="pk", private_key="sk", region="cn-wlcb", zone="cn-wlcb-01")
        self.states = states or ["Install", "Starting", "Running"]
        self.specs = specs or [
            GpuSpecCombination(1, 16, 64, True),
            GpuSpecCombination(2, 32, 128, True),
            GpuSpecCombination(8, 92, 940, False),
        ]
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.stopped: list[str] = []
        self.capacity_zones: list[str | None] = []
        self.describe_calls = 0

    def check_resource_capacity(self, gpu_type=None, image_id=None, charge_type=None, cpu_platform="Intel/Auto", zone=None):  # type: ignore[override]
        from backend.providers.youyun.models import ResourceCapacity

        self.capacity_zones.append(zone)
        return ResourceCapacity(gpu_type=gpu_type or "4090", specs=self.specs)

    def describe_support_zones(self, region=None):  # type: ignore[override]
        # 真实 API 会忽略 Region 过滤，返回全部区域的可用区。
        return [
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
        ]

    def create_instance(self, name, gpu, cpu, memory_gb, image_id=None, gpu_type=None, charge_type=None, zone=None):  # type: ignore[override]
        self.created.append({"name": name, "gpu": gpu, "cpu": cpu, "memory_gb": memory_gb, "zone": zone})
        return ["uhost-1"]

    def describe_instance(self, uhost_id):  # type: ignore[override]
        self.describe_calls += 1
        index = min(self.describe_calls - 1, len(self.states) - 1)
        state = self.states[index]
        if state == "Deleted":
            # 真实 API 对已删除实例返回空列表，client 会抛“实例不存在”
            from backend.providers.youyun.models import YouyunApiError

            raise YouyunApiError("DescribeCompShareInstance", 8, "实例不存在")
        return _make_instance(state=state)

    def delete_instance(self, uhost_id):  # type: ignore[override]
        self.deleted.append(uhost_id)
        self.states = ["Deleted"]
        return None

    def stop_instance(self, uhost_id):  # type: ignore[override]
        self.stopped.append(uhost_id)
        return None


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _manager(client: FakeClient, **overrides) -> YouyunInstanceManager:
    params = dict(poll_interval=0, boot_timeout=30, terminate_timeout=15, sleep=lambda s: None, clock=_FakeClock())
    params.update(overrides)
    return YouyunInstanceManager(client, **params)


class CreateOneShotTest(unittest.TestCase):
    def test_rejects_pod_mode_with_uhost_zone_before_capacity_check(self) -> None:
        client = FakeClient()
        with self.assertRaisesRegex(Exception, "Pod 模式不能使用整机可用区"):
            _manager(client).create_one_shot("t1", instance_mode="pod", zone="cn-wlcb-01")

    def test_rejects_zone_from_different_region_before_capacity_check(self) -> None:
        # 回归：Region 停留在配置默认值而 Zone 属于其他区域时，
        # 优云会以 RetCode=230 (Params [Zone] not available) 拒绝。
        client = FakeClient()
        with self.assertRaisesRegex(Exception, "属于区域 cn-bj2"):
            _manager(client).create_one_shot("t1", instance_mode="pod", zone="cn-bj2-03")
        self.assertEqual(client.capacity_zones, [])

    def test_pod_zone_in_configured_region_reaches_capacity_and_create(self) -> None:
        client = FakeClient()
        instance = _manager(client).create_one_shot("t1", instance_mode="pod", zone="cn-wlcb-03")
        self.assertEqual(instance.uhost_id, "uhost-1")
        self.assertEqual(client.capacity_zones, ["cn-wlcb-03"])
        self.assertEqual(client.created[0]["zone"], "cn-wlcb-03")

    def test_resolves_pod_zone_from_region_when_request_omits_zone(self) -> None:
        client = FakeClient()
        instance = _manager(client).create_one_shot("t1", instance_mode="pod")
        self.assertEqual(instance.uhost_id, "uhost-1")
        self.assertEqual(client.capacity_zones, ["cn-wlcb-03"])
        self.assertEqual(client.created[0]["zone"], "cn-wlcb-03")

    def test_creates_with_smallest_available_spec_and_waits_running(self) -> None:
        client = FakeClient()
        manager = _manager(client)
        instance = manager.create_one_shot("t1")
        self.assertTrue(instance.running)
        self.assertEqual(client.created[0]["gpu"], 1)
        self.assertEqual(client.created[0]["cpu"], 16)
        self.assertEqual(client.created[0]["memory_gb"], 64)
        self.assertEqual(client.created[0]["name"], "cppt-t1")

    def test_respects_explicit_gpu_count(self) -> None:
        client = FakeClient()
        _manager(client).create_one_shot("t1", gpu=2)
        self.assertEqual(client.created[0]["gpu"], 2)

    def test_automatic_selection_falls_back_when_previous_shape_is_unavailable(self) -> None:
        client = FakeClient(specs=[
            GpuSpecCombination(1, 8, 32, False),
            GpuSpecCombination(1, 16, 64, True),
        ])
        _manager(client).create_one_shot(
            "t1",
            gpu=1,
            cpu=8,
            memory_gb=32,
            allow_resource_fallback=True,
        )
        self.assertEqual(client.created[0]["cpu"], 16)
        self.assertEqual(client.created[0]["memory_gb"], 64)

    def test_manual_selection_stays_strict_when_previous_shape_is_unavailable(self) -> None:
        client = FakeClient(specs=[
            GpuSpecCombination(1, 8, 32, False),
            GpuSpecCombination(1, 16, 64, True),
        ])
        with self.assertRaisesRegex(Exception, "所选优云 GPU 规格当前无库存"):
            _manager(client).create_one_shot(
                "t1",
                gpu=1,
                cpu=8,
                memory_gb=32,
            )

    def test_no_inventory_raises(self) -> None:
        client = FakeClient(specs=[GpuSpecCombination(8, 92, 940, False)])
        with self.assertRaises(Exception) as ctx:
            _manager(client).create_one_shot("t1")
        self.assertIn("无可用规格库存", str(ctx.exception))

    def test_fail_state_raises(self) -> None:
        client = FakeClient(states=["Install", "Fail"])
        with self.assertRaises(Exception) as ctx:
            _manager(client).create_one_shot("t1")
        self.assertIn("异常状态", str(ctx.exception))


class CleanupTest(unittest.TestCase):
    def test_terminate_never_raises_and_reports(self) -> None:
        client = FakeClient(states=["Running", "Deleted"])
        manager = _manager(client)
        report = manager.terminate("uhost-1")
        self.assertTrue(report["terminated"])
        self.assertEqual(client.deleted, ["uhost-1"])

    def test_cleanup_after_task_stops_then_deletes(self) -> None:
        client = FakeClient(states=["Running"])
        manager = _manager(client)
        report = manager.cleanup_after_task("uhost-1")
        self.assertTrue(report["terminated"])
        self.assertEqual(client.stopped, ["uhost-1"])
        self.assertEqual(client.deleted, ["uhost-1"])
        self.assertEqual(report["steps"][0]["step"], "stop")

    def test_delete_failure_recorded_not_raised(self) -> None:
        client = FakeClient(states=["Running"])

        def fail_delete(uhost_id):
            from backend.providers.youyun.models import YouyunApiError

            raise YouyunApiError("TerminateCompShareInstance", 230, "欠费无法删除")

        client.delete_instance = fail_delete  # type: ignore[method-assign]
        report = _manager(client).terminate("uhost-1")
        self.assertFalse(report["terminated"])
        self.assertIn("欠费无法删除", str(report["error"]))


class CredentialTest(unittest.TestCase):
    def test_ssh_credentials_decodes_password(self) -> None:
        manager = _manager(FakeClient())
        creds = manager.ssh_credentials(_make_instance())
        self.assertEqual(creds["password"], "password")
        self.assertEqual(creds["port"], 60022)

    def test_decode_password_invalid_returns_none(self) -> None:
        self.assertIsNone(_decode_password("!!!not-base64!!!"))
        self.assertIsNone(_decode_password(""))


if __name__ == "__main__":
    unittest.main()
