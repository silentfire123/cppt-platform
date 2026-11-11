"""优云 Provider 单元测试：签名、参数拍平、实例解析。"""
from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from backend.providers.youyun.client import YouyunClient, _flatten, _resource_params
from backend.providers.youyun.models import YouyunConfig, YouyunInstance, build_region_capabilities
from backend.providers.youyun.ssh import YouyunSSHClient
from backend.providers.youyun.signer import encode_value, signature_base, sign


class SignerTest(unittest.TestCase):
    def test_official_doctest_vector(self) -> None:
        # 官方 SDK 文档样例：verify_ac("my_private_key", {"foo": "bar"})
        signed = sign({"foo": "bar"}, "my_private_key")
        self.assertEqual(signed["Signature"], "634edc1bb957c0d65e5ab5494cf3b7784fbc87af")

    def test_base_is_key_value_concat_then_private_key(self) -> None:
        base = signature_base({"b": "2", "a": "1"}, "sk")
        self.assertEqual(base, "a1b2sk")
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
        self.assertEqual(sign({"b": "2", "a": "1"}, "sk")["Signature"], digest)

    def test_encode_value_special_cases(self) -> None:
        self.assertEqual(encode_value(True), "true")
        self.assertEqual(encode_value(False), "false")
        self.assertEqual(encode_value(2.0), "2")
        self.assertEqual(encode_value(2.5), "2.5")
        self.assertEqual(encode_value("任务 名称"), "任务 名称")
        self.assertEqual(encode_value(100), "100")


class YouyunRegionCapabilitiesTest(unittest.TestCase):
    def test_aggregates_dual_mode_regions_without_conflating_zone_mode(self) -> None:
        capabilities = build_region_capabilities([
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-01", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
        ])

        by_region = {item["region"]: item for item in capabilities}
        self.assertEqual(by_region["cn-wlcb"]["modes"], {"pod": "cn-wlcb-03", "uhost": "cn-wlcb-01"})
        self.assertEqual(by_region["cn-sh2"]["modes"], {"pod": "cn-sh2-01", "uhost": "cn-sh2-02"})
        self.assertEqual(by_region["cn-bj2"]["modes"], {"pod": "cn-bj2-03", "uhost": None})


class YouyunRegionCapabilitiesTest(unittest.TestCase):
    def test_aggregates_dual_mode_regions_without_conflating_zone_mode(self) -> None:
        capabilities = build_region_capabilities([
            {"region": "cn-wlcb", "zone": "cn-wlcb-01", "is_pod": False},
            {"region": "cn-wlcb", "zone": "cn-wlcb-03", "is_pod": True},
            {"region": "cn-sh2", "zone": "cn-sh2-02", "is_pod": False},
            {"region": "cn-sh2", "zone": "cn-sh2-01", "is_pod": True},
            {"region": "cn-bj2", "zone": "cn-bj2-03", "is_pod": True},
        ])

        by_region = {item["region"]: item for item in capabilities}
        self.assertEqual(by_region["cn-wlcb"]["modes"], {"pod": "cn-wlcb-03", "uhost": "cn-wlcb-01"})
        self.assertEqual(by_region["cn-sh2"]["modes"], {"pod": "cn-sh2-01", "uhost": "cn-sh2-02"})
        self.assertEqual(by_region["cn-bj2"]["modes"], {"pod": "cn-bj2-03", "uhost": None})


class YouyunConfigTest(unittest.TestCase):
    def test_training_image_default_is_separate_from_inference_image(self) -> None:
        config = YouyunConfig(image_id="vllm-image")
        self.assertEqual(config.training_image_id, "compshareImage-1bvy5l0kjip2")
        self.assertEqual(config.image_id, "vllm-image")

    def test_pod_configuration_can_use_pod_zone_without_uhost_zone(self) -> None:
        config = YouyunConfig(
            public_key="pk",
            private_key="sk",
            region="cn-wlcb",
            instance_mode="pod",
            pod_zone="cn-wlcb-pod-01",
        )
        self.assertEqual(config.missing_keys(), [])


class FlattenTest(unittest.TestCase):
    def test_flatten_nested_arrays(self) -> None:
        flat = _flatten({"Disks": [{"IsBoot": True, "Size": 100}], "Region": "cn-wlcb"})
        self.assertEqual(
            flat,
            {"Disks.0.IsBoot": True, "Disks.0.Size": 100, "Region": "cn-wlcb"},
        )


class ResourceRequestConsistencyTest(unittest.TestCase):
    def test_capacity_and_create_share_region_zone_gpu_image_and_charge_fields(self) -> None:
        config = YouyunConfig(
            public_key="pk",
            private_key="sk",
            region="cn-wlcb",
            zone="cn-wlcb-01",
            image_id="img-training",
            gpu_type="4090",
            charge_type="Postpay",
        )
        common = _resource_params(
            config,
            gpu_type="A100",
            image_id="img-selected",
            charge_type="Prepay",
            zone="cn-sh2-02",
        )
        self.assertEqual(
            common,
            {
                "Region": "cn-wlcb",
                "Zone": "cn-sh2-02",
                "GpuType": "A100",
                "CompShareImageId": "img-selected",
                "ChargeType": "Prepay",
            },
        )

    def test_create_failure_keeps_provider_action_and_ret_code(self) -> None:
        client = YouyunClient(
            YouyunConfig(
                public_key="pk",
                private_key="sk",
                region="cn-wlcb",
                zone="cn-wlcb-01",
                max_retries=2,
            )
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"RetCode": 226645, "Message": "Service unavailable"}).encode()

        with patch("backend.providers.youyun.client.urllib.request.urlopen", return_value=Response()) as request, patch(
            "backend.providers.youyun.client.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(Exception, "226645.*Service unavailable"):
                client.create_instance("cppt-test", 1, 16, 64, image_id="img", gpu_type="4090")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


class InstanceParseTest(unittest.TestCase):
    def test_parses_ssh_command(self) -> None:
        instance = YouyunInstance.from_api(
            {
                "UHostId": "uhost-1",
                "State": "Running",
                "SshLoginCommand": "ssh root@1.2.3.4 -p 60022",
                "Password": "cGFzc3dvcmQ=",
                "GPU": 1,
                "GpuType": "4090",
                "Memory": 65536,
            }
        )
        self.assertTrue(instance.running)
        self.assertEqual(instance.ip, "1.2.3.4")
        self.assertEqual(instance.ssh_port, 60022)
        self.assertEqual(instance.ssh_user, "root")
        self.assertEqual(instance.memory_mb, 65536)

    def test_without_ssh_info(self) -> None:
        instance = YouyunInstance.from_api({"UHostId": "uhost-2", "State": "Install"})
        self.assertFalse(instance.running)
        self.assertIsNone(instance.ip)
        self.assertEqual(instance.ssh_user, "root")

    def test_parses_port_first_ssh_command(self) -> None:
        # 优云实际返回顺序：ssh -p 23 root@1.2.3.4
        instance = YouyunInstance.from_api({"UHostId": "uhost-3", "SshLoginCommand": "ssh -p 23 root@192.0.2.30"})
        self.assertEqual(instance.ip, "192.0.2.30")
        self.assertEqual(instance.ssh_port, 23)
        self.assertEqual(instance.ssh_user, "root")


class SSHClientTest(unittest.TestCase):
    def test_reconnects_before_command_when_transport_is_inactive(self) -> None:
        class Channel:
            def recv_exit_status(self):
                return 0

        class Stream:
            channel = Channel()

            def read(self):
                return b"ok"

        class Stdin:
            def close(self):
                return None

        class Session:
            def __init__(self, active):
                self.active = active

            def get_transport(self):
                return self

            def is_active(self):
                return self.active

            def exec_command(self, command, timeout):
                return Stdin(), Stream(), Stream()

            def close(self):
                self.active = False

        class Factory:
            def __init__(self):
                self.calls = 0

            def connect(self, **kwargs):
                self.calls += 1
                return Session(True)

        factory = Factory()
        client = YouyunSSHClient(transport=factory)
        client.connect("1.2.3.4", 23, "root", "pass")
        client._client.active = False
        code, output = client.run("echo ok")
        self.assertEqual(code, 0)
        self.assertEqual(output, "okok")
        self.assertEqual(factory.calls, 2)

    def test_reconnects_when_session_dies_during_channel_open(self) -> None:
        class Channel:
            def recv_exit_status(self):
                return 0

        class Stream:
            channel = Channel()

            def read(self):
                return b"ok"

        class Stdin:
            def close(self):
                return None

        class Session:
            def __init__(self, fail):
                self.fail = fail

            def get_transport(self):
                return self

            def is_active(self):
                return True

            def exec_command(self, command, timeout):
                if self.fail:
                    raise RuntimeError("SSH session not active")
                return Stdin(), Stream(), Stream()

            def close(self):
                return None

        class Factory:
            def __init__(self):
                self.calls = 0

            def connect(self, **kwargs):
                self.calls += 1
                return Session(self.calls == 1)

        factory = Factory()
        client = YouyunSSHClient(transport=factory)
        client.connect("1.2.3.4", 23, "root", "pass")
        code, output = client.run("echo ok")
        self.assertEqual(code, 0)
        self.assertEqual(output, "okok")
        self.assertEqual(factory.calls, 2)

    def test_reconnects_when_channel_open_times_out(self) -> None:
        class Channel:
            def recv_exit_status(self):
                return 0

        class Stream:
            channel = Channel()

            def read(self):
                return b"ok"

        class Stdin:
            def close(self):
                return None

        class Session:
            def __init__(self, fail):
                self.fail = fail

            def get_transport(self):
                return self

            def is_active(self):
                return True

            def exec_command(self, command, timeout):
                if self.fail:
                    raise TimeoutError("Timeout opening channel.")
                return Stdin(), Stream(), Stream()

            def close(self):
                return None

        class Factory:
            def __init__(self):
                self.calls = 0

            def connect(self, **kwargs):
                self.calls += 1
                return Session(self.calls == 1)

        factory = Factory()
        client = YouyunSSHClient(transport=factory)
        client.connect("1.2.3.4", 23, "root", "pass")
        code, output = client.run("echo ok")
        self.assertEqual(code, 0)
        self.assertEqual(output, "okok")
        self.assertEqual(factory.calls, 2)

    def test_classifies_paramiko_channel_timeout_as_transient(self) -> None:
        self.assertTrue(YouyunSSHClient._is_transient_connection_error(TimeoutError("timed out")))
        self.assertTrue(
            YouyunSSHClient._is_transient_connection_error(
                RuntimeError("Timeout opening channel.")
            )
        )

    def test_reconnects_when_sftp_channel_reports_server_connection_dropped(self) -> None:
        class Sftp:
            def __init__(self) -> None:
                self.paths: set[str] = set()

            def stat(self, path: str):
                if path not in self.paths:
                    raise FileNotFoundError(path)
                return object()

            def mkdir(self, path: str) -> None:
                self.paths.add(path)

            def close(self) -> None:
                return None

        class Session:
            def __init__(self, fail_sftp: bool) -> None:
                self.fail_sftp = fail_sftp
                self.sftp = Sftp()

            def get_transport(self):
                return self

            def is_active(self):
                return True

            def open_sftp(self):
                if self.fail_sftp:
                    raise RuntimeError("Server connection dropped:")
                return self.sftp

            def close(self) -> None:
                return None

        class Factory:
            def __init__(self) -> None:
                self.calls = 0

            def connect(self, **kwargs):
                self.calls += 1
                return Session(self.calls == 1)

        factory = Factory()
        client = YouyunSSHClient(transport=factory)
        client.connect("1.2.3.4", 23, "root", "pass")
        client.mkdirs("/workspace/runtime/svc-1")

        self.assertEqual(factory.calls, 2)

    def test_reconnects_when_first_sftp_operation_reports_server_connection_dropped(self) -> None:
        class Sftp:
            def __init__(self, fail_stat: bool) -> None:
                self.fail_stat = fail_stat
                self.paths: set[str] = set()

            def stat(self, path: str):
                if self.fail_stat:
                    self.fail_stat = False
                    raise RuntimeError("Server connection dropped:")
                if path not in self.paths:
                    raise FileNotFoundError(path)
                return object()

            def mkdir(self, path: str) -> None:
                self.paths.add(path)

            def close(self) -> None:
                return None

        class Session:
            def __init__(self, fail_stat: bool) -> None:
                self.sftp = Sftp(fail_stat)

            def get_transport(self):
                return self

            def is_active(self):
                return True

            def open_sftp(self):
                return self.sftp

            def close(self) -> None:
                return None

        class Factory:
            def __init__(self) -> None:
                self.calls = 0

            def connect(self, **kwargs):
                self.calls += 1
                return Session(self.calls == 1)

        factory = Factory()
        client = YouyunSSHClient(transport=factory)
        client.connect("1.2.3.4", 23, "root", "pass")
        client.mkdirs("/workspace/runtime/svc-1")

        self.assertEqual(factory.calls, 2)


class DiscoveryTest(unittest.TestCase):
    def _client(self) -> YouyunClient:
        config = YouyunConfig(public_key="pk", private_key="sk", region="cn-wlcb", zone="cn-wlcb-01")
        return YouyunClient(config)

    def test_describe_support_zones(self) -> None:
        client = self._client()
        calls: list[tuple[str, dict]] = []

        def fake_post(action: str, params: dict) -> dict:
            calls.append((action, params))
            return {"RetCode": 0, "ZoneInfo": [{"Region": "cn-wlcb", "Zone": "cn-wlcb-01", "Describe": "乌兰察布一区", "IsPod": False}]}

        client._post = fake_post  # type: ignore[method-assign]
        zones = client.describe_support_zones()
        self.assertEqual(calls[0][0], "DescribeCompShareSupportZone")
        self.assertNotIn("Region", calls[0][1])
        self.assertEqual(zones[0]["zone"], "cn-wlcb-01")
        self.assertFalse(zones[0]["is_pod"])

    def test_describe_images_parses_and_filters_region(self) -> None:
        client = self._client()
        calls: list[tuple[str, dict]] = []

        def fake_post(action: str, params: dict) -> dict:
            calls.append((action, params))
            return {
                "RetCode": 0,
                "ImageSet": [{"CompShareImageId": "img-1", "Name": "CUDA 12.4 PyTorch", "ImageType": "App", "Container": "True", "Author": "优云智算", "ImageOwnerAlias": "Official", "Softwares": {"OsVersion": "Ubuntu 22.04"}, "Zone": "", "ImageSupportZone": None, "Status": "Available", "Tags": ["pytorch"], "Description": ""}],
            }

        client._post = fake_post  # type: ignore[method-assign]
        images = client.describe_images()
        self.assertEqual(calls[0][1]["Region"], "cn-wlcb")
        self.assertEqual(images[0]["image_id"], "img-1")
        self.assertEqual(images[0]["tags"], ["pytorch"])
        self.assertEqual(images[0]["container"], "True")
        self.assertEqual(images[0]["softwares"]["OsVersion"], "Ubuntu 22.04")

    def test_describe_model_repository_models_uses_search_filters_and_maps_paths(self) -> None:
        client = self._client()
        calls: list[tuple[str, dict]] = []

        def fake_post(action: str, params: dict) -> dict:
            calls.append((action, params))
            return {
                "RetCode": 0,
                "Models": [
                    {
                        "Name": "Llama 3 8B",
                        "Path": "/models/llama-3-8b",
                        "Tag": "LLM",
                        "Size": "16GB",
                        "CreateTime": "2026-01-01 00:00:00",
                        "UpdateTime": 1704067200,
                        "ModelId": "model-public-1",
                    },
                    {"Name": "missing-path", "Path": ""},
                ],
            }

        client._post = fake_post  # type: ignore[method-assign]
        models = client.describe_model_repository_models(name=" llama ", tags=" LLM ")
        self.assertEqual(calls, [("DescribeModelRepositoryModels", {"Region": "cn-wlcb", "name": "llama", "tags": "LLM"})])
        self.assertEqual(models, [{
            "name": "Llama 3 8B",
            "path": "/models/llama-3-8b",
            "tag": "LLM",
            "size": "16GB",
            "createTime": "2026-01-01 00:00:00",
            "updateTime": 1704067200,
            "id": "model-public-1",
            "sizeBytes": None,
            "status": None,
            "tags": [],
        }])

    def test_get_instance_price_selects_requested_charge_type(self) -> None:
        client = self._client()
        calls: list[tuple[str, dict]] = []

        def fake_post(action: str, params: dict) -> dict:
            calls.append((action, params))
            return {
                "RetCode": 0,
                "PriceDetails": [
                    {"ChargeType": "Month", "Instance": 1068, "SystemDisks": 120},
                    {"ChargeType": "Postpay", "Instance": 5.2, "SystemDisks": 0.8},
                ],
            }

        client._post = fake_post  # type: ignore[method-assign]
        price = client.get_instance_price(gpu=1, cpu=16, memory_gb=64)
        self.assertEqual(calls[0][0], "GetCompShareInstancePrice")
        self.assertEqual(calls[0][1]["Memory"], 64 * 1024)
        self.assertEqual(price.charge_type, "Postpay")
        self.assertAlmostEqual(price.total, 6.0)


if __name__ == "__main__":
    unittest.main()
