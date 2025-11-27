"""
Test suite for Kafka Maintenance Scheduler (Task 6.3).

Tests automated scheduling and execution of Kafka backend maintenance procedures.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import yaml

from cryptofeed.backends.kafka.maintenance.scheduler import (
    MaintenanceScheduler,
    MaintenanceTask,
    TaskStatus,
    ScheduleConfig,
    MaintenanceExecutor,
)


class TestMaintenanceTask:
    """Test MaintenanceTask data model."""

    def test_task_creation(self):
        """MaintenanceTask should be created with required fields."""
        task = MaintenanceTask(
            task_id="health-check-1",
            task_type="health_check",
            schedule_cron="0 * * * *",  # Every hour
            parameters={"implementation": "modern"},
        )

        assert task.task_id == "health-check-1"
        assert task.task_type == "health_check"
        assert task.schedule_cron == "0 * * * *"
        assert task.parameters["implementation"] == "modern"
        assert task.enabled is True
        assert task.status == TaskStatus.PENDING

    def test_task_status_transitions(self):
        """Task status should transition correctly."""
        task = MaintenanceTask(
            task_id="test-1",
            task_type="analytics_report",
            schedule_cron="0 0 * * *",
        )

        # Pending -> Running
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING

        # Running -> Completed (using mark_completed)
        task.mark_completed()
        assert task.status == TaskStatus.COMPLETED
        assert task.last_run is not None

    def test_task_failure_tracking(self):
        """Task should track failures and retry count."""
        task = MaintenanceTask(
            task_id="test-1",
            task_type="health_check",
            schedule_cron="* * * * *",
        )

        task.status = TaskStatus.FAILED
        task.failure_count += 1
        task.last_error = "Connection timeout"

        assert task.status == TaskStatus.FAILED
        assert task.failure_count == 1
        assert task.last_error == "Connection timeout"


class TestScheduleConfig:
    """Test ScheduleConfig loading and validation."""

    def test_config_loading_from_file(self):
        """ScheduleConfig should load from YAML file."""
        config_data = {
            "tasks": [
                {
                    "task_id": "health-check-legacy",
                    "task_type": "health_check",
                    "schedule_cron": "0 * * * *",
                    "parameters": {"implementation": "legacy"},
                },
                {
                    "task_id": "analytics-daily",
                    "task_type": "analytics_report",
                    "schedule_cron": "0 0 * * *",
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            config = ScheduleConfig.from_yaml(config_file)

            assert len(config.tasks) == 2
            assert config.tasks[0].task_id == "health-check-legacy"
            assert config.tasks[1].task_type == "analytics_report"
        finally:
            Path(config_file).unlink()

    def test_config_validation(self):
        """ScheduleConfig should validate task configurations."""
        config = ScheduleConfig(
            tasks=[
                MaintenanceTask(
                    task_id="test-1",
                    task_type="health_check",
                    schedule_cron="0 * * * *",
                )
            ]
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_detects_invalid_cron(self):
        """ScheduleConfig should detect invalid cron expressions."""
        # Skip if croniter not available (validation only works with croniter)
        try:
            from croniter import croniter
            _ = croniter
            has_croniter = True
        except ImportError:
            has_croniter = False

        if not has_croniter:
            pytest.skip("croniter not installed, cron validation skipped")

        config = ScheduleConfig(
            tasks=[
                MaintenanceTask(
                    task_id="test-1",
                    task_type="health_check",
                    schedule_cron="invalid",  # Invalid cron
                )
            ]
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("cron" in str(e).lower() for e in errors)


class TestMaintenanceExecutor:
    """Test MaintenanceExecutor task execution."""

    def test_executor_health_check_task(self):
        """Executor should execute health_check tasks."""
        executor = MaintenanceExecutor()

        task = MaintenanceTask(
            task_id="health-check-1",
            task_type="health_check",
            schedule_cron="* * * * *",
            parameters={
                "implementation": "modern",
                "bootstrap_servers": ["localhost:9092"],
            },
        )

        # Mock health check
        with patch("cryptofeed.backends.kafka.maintenance.scheduler.MaintenanceCoordinator") as mock_coordinator:
            mock_instance = MagicMock()
            mock_coordinator.return_value = mock_instance
            mock_instance.handle_health_check.return_value = MagicMock(success=True)

            result = executor.execute(task)

            assert result.success is True
            assert task.status == TaskStatus.COMPLETED
            assert task.last_run is not None

    def test_executor_analytics_report_task(self):
        """Executor should execute analytics_report tasks."""
        executor = MaintenanceExecutor()

        task = MaintenanceTask(
            task_id="analytics-1",
            task_type="analytics_report",
            schedule_cron="0 0 * * *",
            parameters={"report_format": "json"},
        )

        with patch("cryptofeed.backends.kafka.maintenance.scheduler.MaintenanceCoordinator") as mock_coordinator:
            mock_instance = MagicMock()
            mock_coordinator.return_value = mock_instance
            mock_instance.get_system_status.return_value = {"status": "ok"}

            result = executor.execute(task)

            assert result.success is True
            assert task.status == TaskStatus.COMPLETED

    def test_executor_deprecation_report_task(self):
        """Executor should execute deprecation_report tasks."""
        executor = MaintenanceExecutor()

        task = MaintenanceTask(
            task_id="deprecation-1",
            task_type="deprecation_report",
            schedule_cron="0 0 * * 0",  # Weekly
        )

        with patch("cryptofeed.backends.kafka.maintenance.scheduler.MaintenanceCoordinator") as mock_coordinator:
            mock_instance = MagicMock()
            mock_coordinator.return_value = mock_instance
            mock_instance.deprecation_monitoring.get_migration_analytics.return_value = {
                "migration_percentage": 50.0
            }

            result = executor.execute(task)

            assert result.success is True

    def test_executor_handles_task_failure(self):
        """Executor should handle task failures gracefully."""
        executor = MaintenanceExecutor()

        task = MaintenanceTask(
            task_id="failing-task",
            task_type="health_check",
            schedule_cron="* * * * *",
        )

        with patch("cryptofeed.backends.kafka.maintenance.scheduler.MaintenanceCoordinator") as mock_coordinator:
            mock_coordinator.side_effect = Exception("Task failed")

            result = executor.execute(task)

            assert result.success is False
            assert task.status == TaskStatus.FAILED
            assert task.failure_count == 1
            assert "Task failed" in task.last_error


class TestMaintenanceScheduler:
    """Test MaintenanceScheduler orchestration."""

    def test_scheduler_initialization(self):
        """Scheduler should initialize with configuration."""
        config_data = {
            "tasks": [
                {
                    "task_id": "test-1",
                    "task_type": "health_check",
                    "schedule_cron": "0 * * * *",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            scheduler = MaintenanceScheduler(config_file)

            assert len(scheduler.tasks) == 1
            assert scheduler.tasks[0].task_id == "test-1"
        finally:
            Path(config_file).unlink()

    def test_scheduler_get_due_tasks(self):
        """Scheduler should identify tasks due for execution."""
        task1 = MaintenanceTask(
            task_id="task-1",
            task_type="health_check",
            schedule_cron="* * * * *",  # Every minute
        )
        # Set next_run to past to make it due
        task1.next_run = datetime.now() - timedelta(minutes=1)

        task2 = MaintenanceTask(
            task_id="task-2",
            task_type="analytics_report",
            schedule_cron="0 * * * *",  # Every hour
        )
        # Set next_run to future to make it not due
        task2.next_run = datetime.now() + timedelta(hours=1)

        ScheduleConfig(tasks=[task1, task2])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"tasks": []}, f)
            config_file = f.name

        try:
            scheduler = MaintenanceScheduler(config_file)
            scheduler.tasks = [task1, task2]

            due_tasks = scheduler.get_due_tasks()

            # Task1 should be due
            assert len(due_tasks) >= 1
            assert task1 in due_tasks
        finally:
            Path(config_file).unlink()

    def test_scheduler_run_once_executes_due_tasks(self):
        """Scheduler should execute all due tasks in run_once."""
        task = MaintenanceTask(
            task_id="test-1",
            task_type="health_check",
            schedule_cron="* * * * *",
            parameters={"bootstrap_servers": ["localhost:9092"]},
        )
        # Make task due
        task.next_run = datetime.now() - timedelta(minutes=1)

        ScheduleConfig(tasks=[task])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"tasks": []}, f)
            config_file = f.name

        try:
            scheduler = MaintenanceScheduler(config_file)
            scheduler.tasks = [task]

            with patch("cryptofeed.backends.kafka.maintenance.scheduler.MaintenanceExecutor.execute") as mock_execute:
                mock_execute.return_value = MagicMock(success=True)

                results = scheduler.run_once()

                assert len(results) >= 1
                mock_execute.assert_called()
        finally:
            Path(config_file).unlink()

    def test_scheduler_status_reporting(self):
        """Scheduler should report task status."""
        task1 = MaintenanceTask(
            task_id="task-1",
            task_type="health_check",
            schedule_cron="* * * * *",
        )
        task1.status = TaskStatus.COMPLETED
        task1.last_run = datetime.now()

        task2 = MaintenanceTask(
            task_id="task-2",
            task_type="analytics_report",
            schedule_cron="0 * * * *",
        )
        task2.status = TaskStatus.FAILED
        task2.failure_count = 3

        ScheduleConfig(tasks=[task1, task2])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"tasks": []}, f)
            config_file = f.name

        try:
            scheduler = MaintenanceScheduler(config_file)
            scheduler.tasks = [task1, task2]

            status = scheduler.get_status()

            assert status["total_tasks"] == 2
            assert status["completed_tasks"] == 1
            assert status["failed_tasks"] == 1
            assert len(status["task_details"]) == 2
        finally:
            Path(config_file).unlink()
