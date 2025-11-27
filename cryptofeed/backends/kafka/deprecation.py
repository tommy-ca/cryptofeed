"""
Deprecation timeline, communication, and decision tracking for Kafka backend evolution.

This module implements Requirements 7.1, 7.2, 7.3, 7.5 from kafka-backend-maintenance spec:
- Timeline management with milestone tracking (7.1, 7.2)
- Multi-channel communication system (7.1, 7.3)
- Decision log maintenance (7.5)
- Progress reporting and usage statistics (7.3)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional
from enum import Enum


class MilestoneStatus(str, Enum):
    """Status values for deprecation milestones."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass
class Milestone:
    """A deprecation timeline milestone with progress tracking."""

    name: str
    description: str
    target_date: datetime
    status: str = "pending"
    completion_percentage: int = 0
    dependencies: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if milestone is complete."""
        return self.completion_percentage == 100 and self.status == "complete"

    @property
    def is_overdue(self) -> bool:
        """Check if milestone is past target date and not complete."""
        return datetime.now() > self.target_date and not self.is_complete

    def mark_complete(self) -> None:
        """Mark milestone as complete."""
        self.status = "complete"
        self.completion_percentage = 100

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "target_date": self.target_date.isoformat(),
            "status": self.status,
            "completion_percentage": self.completion_percentage,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Milestone:
        """Create milestone from dictionary."""
        data = data.copy()
        data["target_date"] = datetime.fromisoformat(data["target_date"])
        return cls(**data)


@dataclass
class ValidationResult:
    """Result of timeline validation."""

    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DeprecationTimeline:
    """
    Manages deprecation timeline with milestones and validation.

    Implements Requirements:
    - 7.1: Multi-channel timeline communication
    - 7.2: Documentation updates at milestones
    """

    def __init__(self, milestones: Optional[Dict[str, Milestone]] = None):
        self.milestones = milestones or {}

    @classmethod
    def load_default(cls) -> DeprecationTimeline:
        """Load the default Kafka backend deprecation timeline."""
        # Base date: Now
        base = datetime.now()

        milestones = {
            "deprecation_warnings": Milestone(
                name="deprecation_warnings",
                description="Implement deprecation warning system for legacy classes",
                target_date=base + timedelta(weeks=2),
                status="complete",
                completion_percentage=100,
            ),
            "migration_tools": Milestone(
                name="migration_tools",
                description="Build configuration migration and validation system",
                target_date=base + timedelta(weeks=4),
                status="complete",
                completion_percentage=100,
                dependencies=["deprecation_warnings"],
            ),
            "documentation": Milestone(
                name="documentation",
                description="Create migration documentation and user guidance",
                target_date=base + timedelta(weeks=6),
                status="in_progress",
                completion_percentage=90,
                dependencies=["migration_tools"],
            ),
            "monitoring": Milestone(
                name="monitoring",
                description="Implement health monitoring and metrics collection",
                target_date=base + timedelta(weeks=8),
                status="complete",
                completion_percentage=100,
                dependencies=["migration_tools"],
            ),
            "shim_removal": Milestone(
                name="shim_removal",
                description="Remove compatibility shim (kafka_callback.py)",
                target_date=base + timedelta(weeks=24),  # Q2 2026 target
                status="pending",
                completion_percentage=0,
                dependencies=["documentation", "monitoring"],
            ),
            "legacy_cleanup": Milestone(
                name="legacy_cleanup",
                description="Remove legacy Kafka backend classes",
                target_date=base + timedelta(weeks=26),
                status="pending",
                completion_percentage=0,
                dependencies=["shim_removal"],
            ),
        }

        return cls(milestones=milestones)

    def validate(self) -> ValidationResult:
        """Validate timeline for overdue milestones and dependency issues."""
        warnings = []
        errors = []

        for name, milestone in self.milestones.items():
            if milestone.is_overdue:
                warnings.append(f"Milestone '{name}' is overdue (target: {milestone.target_date.date()})")

            # Check dependencies
            for dep in milestone.dependencies:
                if dep not in self.milestones:
                    errors.append(f"Milestone '{name}' depends on unknown milestone '{dep}'")
                elif not self.milestones[dep].is_complete and milestone.status == "complete":
                    warnings.append(f"Milestone '{name}' complete but dependency '{dep}' is not")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)

    def to_markdown(self) -> str:
        """Export timeline as markdown table."""
        lines = [
            "# Kafka Backend Deprecation Timeline",
            "",
            "| Phase | Description | Target Date | Status | Progress |",
            "|-------|-------------|-------------|--------|----------|",
        ]

        for name, milestone in self.milestones.items():
            status_emoji = {
                "complete": "✅",
                "in_progress": "🚧",
                "pending": "⏸️",
                "blocked": "🚫",
            }.get(milestone.status, "")

            lines.append(
                f"| {milestone.name} | {milestone.description} | "
                f"{milestone.target_date.date()} | {status_emoji} {milestone.status} | "
                f"{milestone.completion_percentage}% |"
            )

        lines.append("")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save timeline to JSON file."""
        data = {"milestones": {name: m.to_dict() for name, m in self.milestones.items()}}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> DeprecationTimeline:
        """Load timeline from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        milestones = {name: Milestone.from_dict(m) for name, m in data["milestones"].items()}
        return cls(milestones=milestones)


@dataclass
class TimelineUpdate:
    """A timeline update notification."""

    milestone_name: str
    old_status: str
    new_status: str
    message: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "milestone_name": self.milestone_name,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CommunicationResult:
    """Result of sending a timeline update."""

    success: bool
    channels_notified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CommunicationSystem:
    """
    Multi-channel communication system for timeline updates.

    Implements Requirement 7.1: Multi-channel timeline communication.
    """

    def __init__(self):
        self._channels = {
            "documentation": self._send_to_documentation,
            "release_notes": self._send_to_release_notes,
            "deprecation_warnings": self._send_to_warnings,
        }
        self._history: List[TimelineUpdate] = []

    def get_registered_channels(self) -> List[str]:
        """Get list of registered communication channels."""
        return list(self._channels.keys())

    def send_update(self, update: TimelineUpdate, channels: str | List[str]) -> CommunicationResult:
        """
        Send timeline update to specified channels.

        Args:
            update: The update to send
            channels: Channel name(s) or "all"
        """
        if channels == "all":
            target_channels = list(self._channels.keys())
        elif isinstance(channels, str):
            target_channels = [channels]
        else:
            target_channels = channels

        notified = []
        errors = []

        for channel in target_channels:
            if channel not in self._channels:
                errors.append(f"Unknown channel: {channel}")
                continue

            try:
                self._channels[channel](update)
                notified.append(channel)
            except Exception as e:
                errors.append(f"Error sending to {channel}: {e}")

        # Track in history
        self._history.append(update)

        return CommunicationResult(success=len(errors) == 0, channels_notified=notified, errors=errors)

    def get_history(self) -> List[TimelineUpdate]:
        """Get communication history."""
        return self._history.copy()

    def _send_to_documentation(self, update: TimelineUpdate) -> None:
        """Send update to documentation (placeholder)."""
        # In real implementation, this would update docs/kafka/timeline.md
        pass

    def _send_to_release_notes(self, update: TimelineUpdate) -> None:
        """Send update to release notes (placeholder)."""
        # In real implementation, this would update CHANGELOG.md
        pass

    def _send_to_warnings(self, update: TimelineUpdate) -> None:
        """Send update to deprecation warning system (placeholder)."""
        # In real implementation, this would update warning messages
        pass


@dataclass
class DecisionRecord:
    """
    An architectural decision record (ADR).

    Implements Requirement 7.5: Decision log maintenance.
    """

    id: str
    title: str
    status: Literal["proposed", "accepted", "rejected", "deprecated", "superseded"]
    context: str
    decision: str
    consequences: List[str]
    date: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "date": self.date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DecisionRecord:
        """Create from dictionary."""
        data = data.copy()
        data["date"] = datetime.fromisoformat(data["date"])
        return cls(**data)

    def to_markdown(self) -> str:
        """Export as markdown ADR."""
        lines = [
            f"# {self.id}: {self.title}",
            "",
            f"**Date:** {self.date.date()}",
            "",
            f"**Status:** {self.status}",
            "",
            "## Context",
            "",
            self.context,
            "",
            "## Decision",
            "",
            self.decision,
            "",
            "## Consequences",
            "",
        ]

        for consequence in self.consequences:
            lines.append(f"- {consequence}")

        lines.append("")
        return "\n".join(lines)


class DecisionLog:
    """
    Manages architectural decision records for Kafka backend evolution.

    Implements Requirement 7.5: Decision log maintenance.
    """

    def __init__(self):
        self._decisions: Dict[str, DecisionRecord] = {}

    def add_decision(self, decision: DecisionRecord) -> None:
        """Add a decision record."""
        self._decisions[decision.id] = decision

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get a decision by ID."""
        return self._decisions.get(decision_id)

    def list_decisions(self) -> List[DecisionRecord]:
        """List all decisions in chronological order."""
        return sorted(self._decisions.values(), key=lambda d: d.date)

    def to_markdown(self) -> str:
        """Export all decisions as markdown."""
        lines = ["# Kafka Backend Architectural Decisions", "", ""]

        for decision in self.list_decisions():
            lines.append(decision.to_markdown())
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def save_to_directory(self, directory: str) -> None:
        """Save each decision as a separate markdown file."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for decision in self._decisions.values():
            # Create filename from ID and title
            filename = f"{decision.id}-{decision.title.lower().replace(' ', '-')}.md"
            file_path = dir_path / filename

            with open(file_path, "w") as f:
                f.write(decision.to_markdown())


@dataclass
class TimelineRecommendation:
    """Recommendation for timeline adjustment based on usage data."""

    should_extend_timeline: bool
    recommended_extension_days: int
    reason: str


class ProgressReport:
    """
    Tracks usage statistics and migration progress.

    Implements Requirement 7.3: Regular progress updates and usage statistics.
    """

    def __init__(self):
        self._legacy_usage: List[dict] = []
        self._modern_usage: List[dict] = []

    def record_legacy_usage(self, class_name: str, context: dict) -> None:
        """Record usage of legacy Kafka backend."""
        self._legacy_usage.append({"class_name": class_name, "context": context, "timestamp": datetime.now()})

    def record_modern_usage(self, class_name: str, context: dict) -> None:
        """Record usage of modern Kafka backend."""
        self._modern_usage.append({"class_name": class_name, "context": context, "timestamp": datetime.now()})

    def get_legacy_usage_stats(self) -> dict:
        """Get legacy usage statistics."""
        classes_used = set(u["class_name"] for u in self._legacy_usage)
        return {"total_legacy_usage": len(self._legacy_usage), "classes_used": list(classes_used)}

    def get_modern_usage_stats(self) -> dict:
        """Get modern usage statistics."""
        classes_used = set(u["class_name"] for u in self._modern_usage)
        return {"total_modern_usage": len(self._modern_usage), "classes_used": list(classes_used)}

    def get_migration_percentage(self) -> float:
        """Calculate migration completion percentage."""
        total = len(self._legacy_usage) + len(self._modern_usage)
        if total == 0:
            return 0.0
        return (len(self._modern_usage) / total) * 100

    def get_timeline_recommendation(self) -> TimelineRecommendation:
        """
        Generate timeline recommendation based on adoption metrics.

        If migration is slower than expected, recommend timeline extension.
        """
        migration_pct = self.get_migration_percentage()

        # If less than 50% migrated, recommend extension
        if migration_pct < 50:
            # Recommend extending by proportion of incomplete migration
            extension_days = int((100 - migration_pct) * 0.9)  # ~90 days max for 0% migration
            return TimelineRecommendation(
                should_extend_timeline=True,
                recommended_extension_days=extension_days,
                reason=f"Slow adoption rate ({migration_pct:.1f}% migrated). Consider extending timeline.",
            )

        return TimelineRecommendation(
            should_extend_timeline=False, recommended_extension_days=0, reason="Migration on track"
        )

    def to_markdown(self) -> str:
        """Export progress report as markdown."""
        legacy_stats = self.get_legacy_usage_stats()
        modern_stats = self.get_modern_usage_stats()
        migration_pct = self.get_migration_percentage()

        lines = [
            "# Kafka Backend Migration Progress",
            "",
            f"**Migration Percentage:** {migration_pct:.1f}%",
            "",
            "## Legacy Usage",
            f"- Total: {legacy_stats['total_legacy_usage']}",
            f"- Classes: {', '.join(legacy_stats['classes_used'])}",
            "",
            "## Modern Usage",
            f"- Total: {modern_stats['total_modern_usage']}",
            f"- Classes: {', '.join(modern_stats['classes_used'])}",
            "",
        ]

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export progress report as JSON."""
        data = {
            "legacy_usage": self.get_legacy_usage_stats(),
            "modern_usage": self.get_modern_usage_stats(),
            "migration_percentage": self.get_migration_percentage(),
            "timestamp": datetime.now().isoformat(),
        }
        return json.dumps(data, indent=2)
