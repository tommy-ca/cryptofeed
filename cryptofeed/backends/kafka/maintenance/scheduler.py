"""
Automated Kafka Maintenance Scheduler (Task 6.3).

Provides automated scheduling and execution of Kafka backend maintenance tasks
including health checks, analytics reporting, and deprecation monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import yaml

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False
    croniter = None

from cryptofeed.backends.kafka.maintenance.integration import (
    MaintenanceCoordinator,
    IntegrationResult,
)

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a maintenance task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class MaintenanceTask:
    """Configuration for a scheduled maintenance task."""

    task_id: str
    task_type: str  # health_check, analytics_report, deprecation_report, config_migration
    schedule_cron: str  # Cron expression
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    status: TaskStatus = TaskStatus.PENDING
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    max_retries: int = 3

    def __post_init__(self):
        """Calculate next run time."""
        if self.enabled and self.schedule_cron:
            self._update_next_run()

    def _update_next_run(self):
        """Update next run time based on cron schedule."""
        try:
            if HAS_CRONITER:
                base_time = self.last_run if self.last_run else datetime.now()
                cron = croniter(self.schedule_cron, base_time)
                self.next_run = cron.get_next(datetime)
            else:
                # Fallback: simple interval-based scheduling
                # Parse cron as "* * * * *" = every minute, "0 * * * *" = every hour, etc.
                base_time = self.last_run if self.last_run else datetime.now()
                self.next_run = base_time + timedelta(minutes=60)  # Default to hourly
        except Exception as e:
            logger.error(f"Failed to calculate next run for {self.task_id}: {e}")
            self.next_run = None

    def is_due(self, current_time: Optional[datetime] = None) -> bool:
        """Check if task is due for execution."""
        if not self.enabled:
            return False

        if current_time is None:
            current_time = datetime.now()

        if self.next_run is None:
            self._update_next_run()

        return self.next_run is not None and current_time >= self.next_run

    def mark_completed(self):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.last_run = datetime.now()
        self.failure_count = 0
        self.last_error = None
        self._update_next_run()

    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.last_run = datetime.now()
        self.failure_count += 1
        self.last_error = error

        # Disable task if max retries exceeded
        if self.failure_count >= self.max_retries:
            logger.error(
                f"Task {self.task_id} exceeded max retries ({self.max_retries}), disabling"
            )
            self.enabled = False
        else:
            self._update_next_run()


@dataclass
class ScheduleConfig:
    """Configuration for maintenance scheduler."""

    tasks: List[MaintenanceTask] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ScheduleConfig":
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using empty config")
            return cls()

        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        tasks = [
            MaintenanceTask(
                task_id=task_data["task_id"],
                task_type=task_data["task_type"],
                schedule_cron=task_data["schedule_cron"],
                parameters=task_data.get("parameters", {}),
                enabled=task_data.get("enabled", True),
                max_retries=task_data.get("max_retries", 3),
            )
            for task_data in data.get("tasks", [])
        ]

        return cls(tasks=tasks)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        for task in self.tasks:
            # Validate cron expression (only if croniter available)
            if HAS_CRONITER:
                try:
                    croniter(task.schedule_cron)
                except Exception as e:
                    errors.append(f"Invalid cron expression for {task.task_id}: {e}")

            # Validate task type
            valid_types = [
                "health_check",
                "analytics_report",
                "deprecation_report",
                "config_migration",
            ]
            if task.task_type not in valid_types:
                errors.append(
                    f"Invalid task type for {task.task_id}: {task.task_type}"
                )

        return errors


class MaintenanceExecutor:
    """Executes maintenance tasks."""

    def __init__(self):
        """Initialize executor."""
        self.coordinator = None

    def _get_coordinator(self) -> MaintenanceCoordinator:
        """Get or create maintenance coordinator."""
        if self.coordinator is None:
            self.coordinator = MaintenanceCoordinator()
        return self.coordinator

    def execute(self, task: MaintenanceTask) -> IntegrationResult:
        """Execute a maintenance task."""
        logger.info(f"Executing task {task.task_id} (type: {task.task_type})")

        task.status = TaskStatus.RUNNING

        try:
            coordinator = self._get_coordinator()

            # Execute based on task type
            if task.task_type == "health_check":
                result = self._execute_health_check(coordinator, task)
            elif task.task_type == "analytics_report":
                result = self._execute_analytics_report(coordinator, task)
            elif task.task_type == "deprecation_report":
                result = self._execute_deprecation_report(coordinator, task)
            elif task.task_type == "config_migration":
                result = self._execute_config_migration(coordinator, task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update task status
            if result.success:
                task.mark_completed()
                logger.info(f"Task {task.task_id} completed successfully")
            else:
                error_msg = "; ".join(result.errors) if result.errors else "Unknown error"
                task.mark_failed(error_msg)
                logger.error(f"Task {task.task_id} failed: {error_msg}")

            return result

        except Exception as e:
            error_msg = f"Exception executing task: {e}"
            task.mark_failed(error_msg)
            logger.exception(f"Task {task.task_id} failed with exception")

            return IntegrationResult(
                success=False,
                events=[],
                errors=[error_msg],
                warnings=[],
            )

    def _execute_health_check(
        self, coordinator: MaintenanceCoordinator, task: MaintenanceTask
    ) -> IntegrationResult:
        """Execute health check task."""
        implementation = task.parameters.get("implementation", "modern")
        bootstrap_servers = task.parameters.get("bootstrap_servers", ["localhost:9092"])

        return coordinator.handle_health_check(
            bootstrap_servers=bootstrap_servers,
            implementation=implementation,
        )

    def _execute_analytics_report(
        self, coordinator: MaintenanceCoordinator, task: MaintenanceTask
    ) -> IntegrationResult:
        """Execute analytics report task."""
        status = coordinator.get_system_status()

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "system_status": status,
        }

        # Log or store report based on parameters
        report_format = task.parameters.get("report_format", "json")
        output_path = task.parameters.get("output_path")

        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        logger.info(f"Analytics report generated: {report}")

        return IntegrationResult(
            success=True,
            events=[],
            errors=[],
            warnings=[],
        )

    def _execute_deprecation_report(
        self, coordinator: MaintenanceCoordinator, task: MaintenanceTask
    ) -> IntegrationResult:
        """Execute deprecation report task."""
        analytics = coordinator.deprecation_monitoring.get_migration_analytics()

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.task_id,
            "migration_analytics": analytics,
        }

        # Log or store report
        output_path = task.parameters.get("output_path")
        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        logger.info(f"Deprecation report generated: migration at {analytics['migration_percentage']}%")

        return IntegrationResult(
            success=True,
            events=[],
            errors=[],
            warnings=[],
        )

    def _execute_config_migration(
        self, coordinator: MaintenanceCoordinator, task: MaintenanceTask
    ) -> IntegrationResult:
        """Execute configuration migration task."""
        legacy_config = task.parameters.get("legacy_config", {})

        return coordinator.handle_configuration_migration(legacy_config)


class MaintenanceScheduler:
    """Scheduler for automated Kafka maintenance tasks."""

    def __init__(self, config_path: str):
        """
        Initialize scheduler.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = ScheduleConfig.from_yaml(config_path)
        self.executor = MaintenanceExecutor()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")

        self.tasks = self.config.tasks
        logger.info(f"Initialized scheduler with {len(self.tasks)} tasks")

    def get_due_tasks(self, current_time: Optional[datetime] = None) -> List[MaintenanceTask]:
        """Get list of tasks due for execution."""
        if current_time is None:
            current_time = datetime.now()

        return [task for task in self.tasks if task.is_due(current_time)]

    def run_once(self) -> List[IntegrationResult]:
        """
        Run scheduler once, executing all due tasks.

        Returns:
            List of execution results
        """
        due_tasks = self.get_due_tasks()

        if not due_tasks:
            logger.debug("No tasks due for execution")
            return []

        logger.info(f"Executing {len(due_tasks)} due tasks")

        results = []
        for task in due_tasks:
            result = self.executor.execute(task)
            results.append(result)

        return results

    def run_forever(self, check_interval_seconds: int = 60):
        """
        Run scheduler continuously.

        Args:
            check_interval_seconds: Interval between checks for due tasks
        """
        import time

        logger.info(f"Starting scheduler (check interval: {check_interval_seconds}s)")

        try:
            while True:
                self.run_once()
                time.sleep(check_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        total_tasks = len(self.tasks)
        enabled_tasks = sum(1 for t in self.tasks if t.enabled)
        completed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        pending_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)

        task_details = [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status.value,
                "enabled": t.enabled,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "next_run": t.next_run.isoformat() if t.next_run else None,
                "failure_count": t.failure_count,
                "last_error": t.last_error,
            }
            for t in self.tasks
        ]

        return {
            "total_tasks": total_tasks,
            "enabled_tasks": enabled_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": pending_tasks,
            "task_details": task_details,
        }
