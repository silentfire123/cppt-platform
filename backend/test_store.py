from __future__ import annotations

import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path

from backend.postgres_storage import PostgresTaskStore
from backend.postgres_storage import MIGRATIONS
from backend.store import (
    TOTAL_STEPS_PATTERN,
    TaskStore,
    infer_metric_step,
    next_task_clone_name,
    parse_trainer_values,
    resolve_adapter_artifact,
    task_clone_base_name,
)


def payload() -> dict:
    return {
        "name": "store test",
        "model": "Qwen/Qwen3-1.7B",
        "datasetId": "qwenruozhi_v1",
        "datasetName": "dataset",
        "parameters": {"epochs": 1},
    }


class TaskStoreTest(unittest.TestCase):
    def test_task_clone_names_use_smallest_free_number(self) -> None:
        self.assertEqual(next_task_clone_name("jcy-test5", []), "jcy-test5 副本1")
        self.assertEqual(
            next_task_clone_name("jcy-test5", ["jcy-test5 副本1"]),
            "jcy-test5 副本2",
        )
        # 已有 副本1/副本3 时补最小空位 副本2
        self.assertEqual(
            next_task_clone_name("jcy-test5", ["jcy-test5 副本1", "jcy-test5 副本3"]),
            "jcy-test5 副本2",
        )
        # 克隆克隆时去掉旧后缀，不再叠加「副本 副本」
        self.assertEqual(
            next_task_clone_name("jcy-test5 副本", ["jcy-test5 副本1", "jcy-test5 副本2"]),
            "jcy-test5 副本3",
        )
        self.assertEqual(task_clone_base_name("jcy-test5 副本 副本"), "jcy-test5")

    def test_task_clone_persists_numbered_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            first = store.create(payload())
            clone1 = store.clone(first["id"])
            clone2 = store.clone(first["id"])
            clone_of_clone = store.clone(clone1["id"])
            self.assertEqual(clone1["name"], "store test 副本1")
            self.assertEqual(clone2["name"], "store test 副本2")
            self.assertEqual(clone_of_clone["name"], "store test 副本3")

    def test_list_page_supports_filters_and_empty_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            first = store.create({**payload(), "name": "alpha"})
            second = store.create({**payload(), "name": "beta"})
            store.update(second["id"], state="failed")
            result = store.list_page(page=1, page_size=1, state="failed", query="beta")
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["totalPages"], 1)
            self.assertEqual(result["items"][0]["id"], second["id"])
            empty = store.list_page(page=2, page_size=1, query="missing")
            self.assertEqual(empty["items"], [])
            self.assertEqual(empty["total"], 0)
    def test_total_optimization_steps_log_is_recognized(self) -> None:
        match = TOTAL_STEPS_PATTERN.search("Total optimization steps = 10")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "10")

    def test_trainer_log_metrics_are_parsed_and_backfilled(self) -> None:
        values = parse_trainer_values(
            "{'loss': '1.234', 'learning_rate': '2e-05', 'epoch': '0.2'}"
        )
        self.assertEqual(values["loss"], 1.234)
        self.assertEqual(
            infer_metric_step(
                values,
                total_steps=100,
                total_epochs=2,
                last_train_step=1,
                logging_steps=10,
            ),
            10,
        )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tasks.db"
            store = TaskStore(database)
            created = store.create(payload())
            store.append_log(created["id"], "Num Epochs = 2 | Total steps = 100")
            store.append_log(
                created["id"],
                "{'loss': '2.5', 'learning_rate': '0', 'epoch': '0.02'}",
            )
            store.append_log(
                created["id"],
                "{'loss': '1.5', 'learning_rate': '2e-05', 'epoch': '0.2'}",
            )
            store.append_log(
                created["id"], "{'eval_loss': '1.75', 'epoch': '0.2'}"
            )
            store.update(created["id"], loss=1.8)

            reopened = TaskStore(database)
            metrics = reopened.get(created["id"])["lossHistory"]
            self.assertEqual([metric["step"] for metric in metrics], [1, 10])
            self.assertEqual(metrics[1]["trainLoss"], 1.5)
            self.assertEqual(metrics[1]["evalLoss"], 1.75)

    def test_final_train_loss_summary_is_parsed(self) -> None:
        values = parse_trainer_values(
            "{'train_runtime': 6.0105, 'train_loss': 3.6688139779227122, 'epoch': 0.93}"
        )

        self.assertEqual(values["loss"], 3.6688139779227122)
        self.assertEqual(values["epoch"], 0.93)

    def test_lightweight_task_list_skips_logs_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create(payload())
            store.append_log(created["id"], "a large worker log")
            store.append_metric(created["id"], step=1, train_loss=1.2)

            summary = store.list(include_details=False)[0]
            self.assertNotIn("logs", summary)
            self.assertEqual(summary["lossHistory"], [])
            detail = store.get(created["id"])
            self.assertEqual(detail["logs"][-1].split("] ", 1)[1], "a large worker log")
            self.assertEqual(len(detail["lossHistory"]), 1)

    def test_dataset_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create_dataset(
                {
                    "id": "ds-test",
                    "name": "dataset test",
                    "type": "文本生成",
                    "structure": "SFT",
                    "state": "ready",
                    "sampleCount": 12,
                    "paths": {"trainPath": "/tmp/train.jsonl"},
                    "validationMessage": "ok",
                    "owner": "tester",
                    "sourceType": "minio",
                    "sourceUri": "s3://bucket/exports/train.jsonl",
                    "sourceBucket": "bucket",
                    "sourceKey": "exports/train.jsonl",
                    "sizeBytes": 123,
                    "contentSha256": "a" * 64,
                }
            )
            self.assertEqual(created["count"], "12 条")
            self.assertEqual(created["sourceType"], "minio")
            self.assertEqual(created["sourceKey"], "exports/train.jsonl")
            versions = store.list_dataset_versions("ds-test")
            self.assertEqual(versions[0]["version"], 1)
            self.assertEqual(versions[0]["content_sha256"], "a" * 64)
            self.assertEqual(store.list_datasets()[0]["state"], "ready")
            updated = store.create_dataset(
                {
                    **created,
                    "state": "failed",
                    "warningCount": 2,
                    "validationMessage": "bad",
                }
            )
            self.assertEqual(updated["state"], "failed")
            repaired = store.create_dataset(
                {
                    **updated,
                    "state": "ready",
                    "warningCount": 0,
                    "validationMessage": "修复完成，格式校验通过，可以发布。",
                }
            )
            self.assertEqual(repaired["state"], "ready")
            self.assertEqual(repaired["warningCount"], 0)
            published = store.update_dataset_state("ds-test", "published")
            self.assertEqual(published["state"], "published")
            deleted = store.delete_dataset("ds-test")
            self.assertEqual(deleted["id"], "ds-test")
            self.assertIsNone(store.get_dataset("ds-test"))

    def test_postgres_migration_includes_dataset_versions(self) -> None:
        migration_sql = "\n".join(sql for _, sql in MIGRATIONS)
        self.assertIn("CREATE TABLE IF NOT EXISTS datasets", migration_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS dataset_versions", migration_sql)
        self.assertIn("UNIQUE(dataset_id, version)", migration_sql)

    def test_postgres_dataset_serializer_exposes_current_version_source(self) -> None:
        row = {
            "id": "ds-pg",
            "name": "postgres dataset",
            "dataset_type": "文本生成",
            "structure": "SFT",
            "state": "ready",
            "sample_count": 3,
            "validation_message": "ok",
            "warning_count": 0,
            "owner": "tester",
            "builtin": False,
            "updated_at": "2026-08-24T10:00:00+08:00",
            "paths": '{"trainPath":"/workspace/datasets/ds-pg/train.jsonl"}',
            "version": 1,
            "source_type": "minio",
            "source_uri": "s3://bucket/exports/train.jsonl",
            "source_bucket": "bucket",
            "source_key": "exports/train.jsonl",
            "size_bytes": 42,
            "content_sha256": "b" * 64,
        }
        dataset = PostgresTaskStore._serialize_dataset(row)
        self.assertEqual(dataset["version"], 1)
        self.assertEqual(dataset["sourceType"], "minio")
        self.assertEqual(dataset["paths"]["trainPath"], "/workspace/datasets/ds-pg/train.jsonl")

    def test_postgres_dataset_create_writes_catalog_and_version_rows(self) -> None:
        class Cursor:
            def __init__(self, row=None):
                self.row = row

            def fetchone(self):
                return self.row

        class Connection:
            def __init__(self):
                self.calls = []

            def execute(self, sql, params=()):
                self.calls.append((sql, params))
                return Cursor(None)

        connection = Connection()
        store = object.__new__(PostgresTaskStore)
        store._lock = threading.RLock()

        @contextmanager
        def connect():
            yield connection

        store._connect = connect
        store.upsert_dataset(
            {
                "id": "ds-pg",
                "name": "dataset",
                "type": "文本生成",
                "structure": "SFT",
                "state": "ready",
                "sampleCount": 2,
                "paths": {"trainPath": "/workspace/train.jsonl"},
                "validationMessage": "ok",
                "sourceType": "minio",
                "sourceBucket": "bucket",
                "sourceKey": "exports/train.jsonl",
                "sourceUri": "s3://bucket/exports/train.jsonl",
                "sizeBytes": 20,
                "contentSha256": "c" * 64,
            }
        )

        self.assertEqual(len(connection.calls), 3)
        self.assertIn("INSERT INTO datasets", connection.calls[1][0])
        self.assertIn("INSERT INTO dataset_versions", connection.calls[2][0])
        version_params = connection.calls[2][1]
        self.assertEqual(version_params[3], "minio")
        self.assertEqual(version_params[6], "exports/train.jsonl")
        self.assertEqual(version_params[9], "c" * 64)

    def test_task_lifecycle_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create(payload())
            self.assertEqual(created["state"], "queued")
            self.assertEqual(created["method"], "SFT / QLoRA")
            self.assertEqual(created["tuningMode"], "qlora")

            claimed = store.claim_next()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["state"], "running")

            store.append_metric(
                created["id"],
                step=1,
                epoch=0.5,
                train_loss=2.5,
                learning_rate=0.0002,
            )
            store.append_metric(created["id"], step=1, eval_loss=2.75)
            metric = store.get(created["id"])["lossHistory"][0]
            self.assertEqual(metric["trainLoss"], 2.5)
            self.assertEqual(metric["evalLoss"], 2.75)

            store.update(
                created["id"],
                state="failed",
                current_step="failed",
                error="test error",
            )
            retried = store.retry(created["id"])
            self.assertIsNotNone(retried)
            self.assertEqual(retried["state"], "queued")
            self.assertIsNone(retried["error"])
            self.assertEqual(len(retried["logs"]), 1)
            self.assertEqual(retried["lossHistory"], [])

            claimed_again = store.claim_next()
            self.assertEqual(claimed_again["id"], created["id"])
            artifacts = [{"type": "adapter", "name": "test", "path": "/tmp/test"}]
            store.update(
                created["id"],
                state="completed",
                progress=100,
                current_step="done",
                loss=1.25,
                artifacts=artifacts,
            )
            completed = store.get(created["id"])
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["artifacts"], artifacts)

    def test_request_stop_marks_running_task_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task = store.create(payload())
            claimed = store.claim_next()
            self.assertEqual(claimed["state"], "running")
            stopping = store.request_stop(task["id"])
            self.assertIsNotNone(stopping)
            assert stopping is not None
            self.assertEqual(stopping["state"], "stopping")
            self.assertTrue(store.is_cancel_requested(task["id"]))

    def test_max_steps_is_serialized_in_task_api_shape(self) -> None:
        values = payload()
        values["maxSteps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create(values)
            self.assertEqual(created["maxSteps"], 1)
            self.assertEqual(store.get(created["id"])["maxSteps"], 1)
            self.assertEqual(store.list()[0]["maxSteps"], 1)

    def test_remote_provider_is_preserved_for_worker_dispatch(self) -> None:
        values = payload()
        values.update({"provider": "youyun", "executionMode": "remote"})
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create(values)
            self.assertEqual(created["provider"], "youyun")
            self.assertEqual(created["executionMode"], "remote")
            self.assertEqual(created["currentStep"], "等待优云 GPU 资源")
            self.assertIn("优云 GPU 队列", created["logs"][-1])
            claimed = store.claim_next()
            self.assertEqual(claimed["provider"], "youyun")
            self.assertEqual(claimed["executionMode"], "remote")
            self.assertEqual(claimed["currentStep"], "启动优云 GPU 训练")

    def test_claim_next_can_select_remote_behind_queued_local_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            local = store.create(payload())
            remote_payload = payload()
            remote_payload.update(
                {
                    "name": "remote",
                    "provider": "youyun",
                    "executionMode": "remote",
                }
            )
            remote = store.create(remote_payload)

            claimed_remote = store.claim_next(provider="youyun")
            assert claimed_remote is not None
            self.assertEqual(claimed_remote["id"], remote["id"])
            claimed_local = store.claim_next(provider="local")
            assert claimed_local is not None
            self.assertEqual(claimed_local["id"], local["id"])

    def test_remote_task_cannot_be_deleted_before_instance_release(self) -> None:
        values = payload()
        values.update({"provider": "youyun", "executionMode": "remote"})
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task = store.create(values)
            store.update(
                task["id"],
                state="completed",
                artifacts=[
                    {
                        "type": "youyun-instance",
                        "instanceId": "uhost-1",
                        "state": "awaiting_comparison",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "尚未释放"):
                store.delete(task["id"])
            current = store.get(task["id"])
            assert current is not None
            store.update(
                task["id"],
                artifacts=[{**current["artifacts"][0], "state": "released"}],
            )
            self.assertTrue(store.delete(task["id"]))

    def test_remote_task_cannot_retry_before_instance_release(self) -> None:
        values = payload()
        values.update({"provider": "youyun", "executionMode": "remote"})
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task = store.create(values)
            store.update(
                task["id"],
                state="stopped",
                artifacts=[
                    {
                        "type": "youyun-instance",
                        "instanceId": "uhost-1",
                        "state": "awaiting_comparison",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "尚未释放"):
                store.retry(task["id"])
            current = store.get(task["id"])
            assert current is not None
            store.update(
                task["id"],
                artifacts=[{**current["artifacts"][0], "state": "released"}],
            )
            retried = store.retry(task["id"])
            assert retried is not None
            self.assertEqual(retried["state"], "queued")

    def test_dpo_task_type_is_serialized(self) -> None:
        values = payload()
        values.update({"taskType": "DPO", "name": "dpo store test"})
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            created = store.create(values)
            self.assertEqual(created["taskType"], "DPO")
            self.assertEqual(created["method"], "DPO / QLoRA")

    def test_lora_mode_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            values = payload()
            values["tuningMode"] = "lora"
            task = store.create(values)
            self.assertEqual(task["method"], "SFT / LoRA")
            self.assertEqual(task["tuningMode"], "lora")

    def test_best_and_final_adapter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task_id = store.create(payload())["id"]
            store.append_metric(task_id, step=10, eval_loss=1.4)
            store.append_metric(task_id, step=20, eval_loss=1.1)
            store.append_metric(task_id, step=30, eval_loss=1.3)
            artifacts = [
                {
                    "type": "adapter",
                    "name": "best",
                    "path": "/tmp/best",
                    "variant": "best",
                    "selected": True,
                    "checkpointStep": 20,
                    "evalLoss": 1.1,
                },
                {
                    "type": "adapter",
                    "name": "final",
                    "path": "/tmp/final",
                    "variant": "final",
                    "selected": False,
                    "checkpointStep": 30,
                    "evalLoss": 1.3,
                },
            ]
            store.update(task_id, state="completed", artifacts=artifacts)

            completed = store.get(task_id)
            self.assertEqual(completed["bestStep"], 20)
            self.assertEqual(completed["bestEvalLoss"], 1.1)
            self.assertEqual(completed["finalEvalLoss"], 1.3)
            self.assertEqual(completed["finalStep"], 30)
            self.assertEqual(completed["selectedAdapterPath"], "/tmp/best")
            self.assertEqual(completed["selectedAdapterVariant"], "best")
            self.assertEqual(completed["bestAdapterPath"], "/tmp/best")
            self.assertEqual(completed["finalAdapterPath"], "/tmp/final")
            self.assertEqual(completed["selectionReason"], "验证集 Loss 最低")
            self.assertEqual(
                resolve_adapter_artifact(artifacts, "final")["path"],
                "/tmp/final",
            )

    def test_legacy_adapter_falls_back_to_final_without_hiding_best_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task_id = store.create(payload())["id"]
            store.append_metric(task_id, step=10, eval_loss=1.1)
            store.append_metric(task_id, step=30, eval_loss=1.3)
            legacy_artifacts = [
                {"type": "adapter", "name": "legacy", "path": "/tmp/legacy"}
            ]
            store.update(task_id, state="completed", artifacts=legacy_artifacts)

            completed = store.get(task_id)
            self.assertEqual(completed["bestStep"], 10)
            self.assertEqual(completed["bestEvalLoss"], 1.1)
            self.assertEqual(completed["finalEvalLoss"], 1.3)
            self.assertEqual(completed["selectedAdapterVariant"], "final")
            self.assertIsNone(completed["bestAdapterPath"])
            self.assertEqual(completed["finalAdapterPath"], "/tmp/legacy")
            self.assertEqual(
                completed["selectionReason"],
                "旧任务未保留历史最佳权重，使用最终 Adapter",
            )
            self.assertIsNone(resolve_adapter_artifact(legacy_artifacts, "best"))
            self.assertEqual(
                resolve_adapter_artifact(legacy_artifacts, "final")["path"],
                "/tmp/legacy",
            )

    def test_artifact_metadata_is_authoritative_for_best_and_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task = store.create(payload())
            store.update(
                task["id"],
                state="completed",
                artifacts=[
                    {
                        "type": "adapter",
                        "name": "best",
                        "path": "/root/artifacts/best",
                        "variant": "best",
                        "selected": True,
                        "checkpointStep": 3,
                        "evalLoss": 8.96159839630127,
                    },
                    {
                        "type": "adapter",
                        "name": "final",
                        "path": "/root/artifacts/final",
                        "variant": "final",
                        "checkpointStep": 3,
                        "evalLoss": 8.96159839630127,
                    },
                ],
            )
            store.append_metric(task["id"], step=2, eval_loss=8.962)
            completed = store.get(task["id"])
            assert completed is not None
            self.assertEqual(completed["bestStep"], 3)
            self.assertAlmostEqual(completed["bestEvalLoss"], 8.96159839630127)
            self.assertEqual(completed["finalStep"], 3)
            self.assertAlmostEqual(completed["finalEvalLoss"], 8.96159839630127)

    def test_completed_training_task_syncs_to_model_catalog_with_real_adapter_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "training-work" / "ft-test" / "output"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            model_cache = root / "models"

            store = TaskStore(root / "tasks.db")
            task = store.create(payload())
            artifacts = [
                {
                    "type": "adapter",
                    "name": "final",
                    "path": str(adapter),
                    "variant": "final",
                    "selected": True,
                    "checkpointStep": 1,
                }
            ]
            store.update(task["id"], state="completed", artifacts=artifacts)
            store.sync_model_assets(model_cache)

            trained = store.get_model_asset(f"trained-{task['id']}")
            self.assertIsNotNone(trained)
            assert trained is not None
            self.assertEqual(trained["kind"], "trained")
            self.assertEqual(trained["status"], "available")
            self.assertEqual(trained["path"], str(adapter.parent))
            self.assertEqual(trained["variants"][0]["path"], str(adapter))
            self.assertEqual(trained["variants"][0]["status"], "available")

    def test_model_asset_page_supports_kind_query_and_archive_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TaskStore(root / "tasks.db")
            task_ids = []
            for name in ["alpha", "beta", "gamma"]:
                task = store.create({**payload(), "name": name})
                store.update(
                    task["id"],
                    state="completed",
                    artifacts=[
                        {
                            "type": "adapter",
                            "name": "final",
                            "path": str(root / name),
                            "variant": "final",
                            "selected": True,
                        }
                    ],
                )
                task_ids.append(task["id"])
            store.sync_model_assets(root / "models")

            first_page = store.list_model_assets_page(
                page=1, page_size=2, kind="trained"
            )
            self.assertEqual(first_page["total"], 3)
            self.assertEqual(first_page["totalPages"], 2)
            self.assertEqual(len(first_page["items"]), 2)

            filtered = store.list_model_assets_page(
                page=1, page_size=5, kind="trained", query="beta"
            )
            self.assertEqual(filtered["total"], 1)
            self.assertEqual(filtered["items"][0]["name"], "beta")

            store.update_model_asset(f"trained-{task_ids[0]}", archived=True)
            archived = store.list_model_assets_page(
                page=1, page_size=5, kind="trained", archived_only=True
            )
            self.assertEqual(archived["total"], 1)
            self.assertTrue(archived["items"][0]["archived"])

    def test_postgres_sync_reads_training_rows_and_delegates_to_catalog(self) -> None:
        class FakeCursor:
            def fetchall(self):
                return [{"id": "ft-pg", "payload": "{}", "state": "completed", "artifacts": "[]", "created_at": "now", "updated_at": "now"}]

        class FakeConnection:
            def __init__(self):
                self.sql = ""

            def execute(self, sql):
                self.sql = sql
                return FakeCursor()

        class FakeContext:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                return self.connection

            def __exit__(self, *args):
                return False

        class FakeCatalog:
            def __init__(self):
                self.rows = None
                self.root = None

            def _sync_model_asset_rows(self, rows, model_cache_root):
                self.rows = rows
                self.root = model_cache_root

        connection = FakeConnection()
        catalog = FakeCatalog()
        store = object.__new__(PostgresTaskStore)
        store.catalog = catalog
        store._connect = lambda: FakeContext(connection)
        model_cache = Path("/tmp/model-cache")

        store.sync_model_assets(model_cache)

        self.assertIn("FROM training_tasks", connection.sql)
        self.assertIn("payload::text", connection.sql)
        self.assertEqual(catalog.rows[0]["id"], "ft-pg")
        self.assertEqual(catalog.root, model_cache)

    def test_postgres_task_list_page_reads_durable_store(self) -> None:
        class FakeCursor:
            def __init__(self, rows):
                self.rows = rows

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return self.rows

        class FakeConnection:
            def __init__(self):
                self.sql = []

            def execute(self, sql, params=()):
                self.sql.append(sql)
                if "COUNT(*)" in sql:
                    return FakeCursor([{"total_count": 1}])
                return FakeCursor([{"id": "ft-pg"}])

        class FakeContext:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                return self.connection

            def __exit__(self, *args):
                return False

        connection = FakeConnection()
        store = object.__new__(PostgresTaskStore)
        store._connect = lambda: FakeContext(connection)
        store._task_query = lambda: "SELECT id FROM training_tasks"
        store._serialize = lambda row, **kwargs: {"id": row["id"]}

        result = store.list_page(page=1, page_size=20, query="pg")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"], [{"id": "ft-pg"}])
        self.assertTrue(all("training_tasks" in sql for sql in connection.sql))
    def test_no_validation_task_selects_final_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task_id = store.create(payload())["id"]
            artifacts = [
                {
                    "type": "adapter",
                    "name": "final",
                    "path": "/tmp/final",
                    "variant": "final",
                    "selected": True,
                    "checkpointStep": 7,
                }
            ]
            store.update(task_id, state="completed", artifacts=artifacts)

            completed = store.get(task_id)
            self.assertIsNone(completed["bestStep"])
            self.assertIsNone(completed["bestEvalLoss"])
            self.assertIsNone(completed["finalEvalLoss"])
            self.assertEqual(completed["finalStep"], 7)
            self.assertEqual(completed["selectedAdapterVariant"], "final")
            self.assertEqual(
                completed["selectionReason"],
                "未产生有效验证 Loss，使用最终 Adapter",
            )

    def test_train_loss_best_adapter_is_exposed_for_no_validation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TaskStore(Path(directory) / "tasks.db")
            task_id = store.create(payload())["id"]
            artifacts = [
                {
                    "type": "adapter",
                    "name": "best",
                    "path": "/tmp/best",
                    "variant": "best",
                    "selected": True,
                    "checkpointStep": 4,
                    "loss": 0.8,
                    "trainLoss": 0.8,
                    "metric": "train_loss",
                },
                {
                    "type": "adapter",
                    "name": "final",
                    "path": "/tmp/final",
                    "variant": "final",
                    "selected": False,
                    "checkpointStep": 7,
                    "loss": 1.1,
                    "trainLoss": 1.1,
                    "metric": "train_loss",
                },
            ]
            store.update(task_id, state="stopped", artifacts=artifacts, stop_reason="manual")

            stopped = store.get(task_id)
            self.assertEqual(stopped["selectionMetric"], "train_loss")
            self.assertEqual(stopped["bestLoss"], 0.8)
            self.assertEqual(stopped["finalLoss"], 1.1)
            self.assertEqual(stopped["bestAdapterPath"], "/tmp/best")
            self.assertEqual(stopped["stopReason"], "manual")
            self.assertEqual(stopped["selectionReason"], "训练 Loss 最低")


if __name__ == "__main__":
    unittest.main()
