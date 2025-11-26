"""
Documentation versioning and rollback system.

This module provides snapshot-based versioning for documentation,
enabling safe rollback when documentation updates cause issues.
"""

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


LOG = logging.getLogger("feedhandler")


class DocumentationVersionManager:
    """
    Manage documentation snapshots and rollback capabilities.

    Provides version control for documentation through snapshots,
    enabling safe rollback when updates introduce errors.
    """

    def __init__(self, snapshot_dir: Optional[Path] = None):
        """
        Initialize the documentation version manager.

        Args:
            snapshot_dir: Directory to store snapshots (defaults to temp directory)
        """
        if snapshot_dir is None:
            # Use a persistent temp directory for snapshots
            snapshot_dir = Path(tempfile.gettempdir()) / "cryptofeed_doc_snapshots"

        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        LOG.debug(f"DocumentationVersionManager initialized with snapshot_dir: {self.snapshot_dir}")

    def create_snapshot(self, docs_dir: Path, tag: str) -> str:
        """
        Create a snapshot of current documentation state.

        Args:
            docs_dir: Directory containing documentation to snapshot
            tag: Tag for this snapshot (e.g., version number)

        Returns:
            Snapshot ID (hash-based identifier)
        """
        docs_dir = Path(docs_dir)

        # Generate snapshot ID from timestamp and tag
        timestamp = datetime.now().isoformat()
        snapshot_id = hashlib.sha256(f"{tag}:{timestamp}".encode()).hexdigest()[:12]

        # Create snapshot directory
        snapshot_path = self.snapshot_dir / snapshot_id
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Copy documentation files
        file_count = 0
        for item in docs_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(docs_dir)
                dest_path = snapshot_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)
                file_count += 1

        # Store metadata
        metadata = {
            "snapshot_id": snapshot_id,
            "tag": tag,
            "timestamp": timestamp,
            "file_count": file_count,
            "source_dir": str(docs_dir)
        }

        metadata_file = snapshot_path / ".snapshot_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        LOG.info(f"Created snapshot {snapshot_id} (tag: {tag}) with {file_count} files")
        return snapshot_id

    def snapshot_exists(self, snapshot_id: str) -> bool:
        """
        Check if a snapshot exists.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            True if snapshot exists
        """
        snapshot_path = self.snapshot_dir / snapshot_id
        return snapshot_path.exists() and (snapshot_path / ".snapshot_metadata.json").exists()

    def get_snapshot_metadata(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Get metadata for a snapshot.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            Dictionary with snapshot metadata

        Raises:
            ValueError: If snapshot doesn't exist
        """
        if not self.snapshot_exists(snapshot_id):
            raise ValueError(f"Snapshot {snapshot_id} does not exist")

        metadata_file = self.snapshot_dir / snapshot_id / ".snapshot_metadata.json"
        return json.loads(metadata_file.read_text(encoding='utf-8'))

    def rollback_to_snapshot(self, snapshot_id: str, target_dir: Path) -> bool:
        """
        Rollback documentation to a previous snapshot.

        Args:
            snapshot_id: Snapshot identifier to rollback to
            target_dir: Target directory to restore documentation

        Returns:
            True if rollback successful

        Raises:
            ValueError: If snapshot doesn't exist
        """
        if not self.snapshot_exists(snapshot_id):
            raise ValueError(f"Snapshot {snapshot_id} does not exist")

        snapshot_path = self.snapshot_dir / snapshot_id
        target_dir = Path(target_dir)

        try:
            # Clear target directory (except hidden files)
            for item in target_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir() and not item.name.startswith('.'):
                    shutil.rmtree(item)

            # Restore files from snapshot
            for item in snapshot_path.rglob("*"):
                if item.is_file() and item.name != ".snapshot_metadata.json":
                    rel_path = item.relative_to(snapshot_path)
                    dest_path = target_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)

            metadata = self.get_snapshot_metadata(snapshot_id)
            LOG.info(f"Rolled back to snapshot {snapshot_id} (tag: {metadata['tag']})")
            return True

        except Exception as e:
            LOG.error(f"Rollback failed: {e}")
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots.

        Returns:
            List of snapshot metadata dictionaries
        """
        snapshots = []

        for snapshot_dir in self.snapshot_dir.iterdir():
            if snapshot_dir.is_dir():
                metadata_file = snapshot_dir / ".snapshot_metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                        snapshots.append(metadata)
                    except Exception as e:
                        LOG.warning(f"Could not read metadata for {snapshot_dir}: {e}")

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        LOG.debug(f"Found {len(snapshots)} snapshots")
        return snapshots

    def diff_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """
        Show differences between two snapshots.

        Args:
            snapshot_id1: First snapshot identifier
            snapshot_id2: Second snapshot identifier

        Returns:
            Dictionary describing differences between snapshots

        Raises:
            ValueError: If either snapshot doesn't exist
        """
        if not self.snapshot_exists(snapshot_id1):
            raise ValueError(f"Snapshot {snapshot_id1} does not exist")
        if not self.snapshot_exists(snapshot_id2):
            raise ValueError(f"Snapshot {snapshot_id2} does not exist")

        snapshot1_path = self.snapshot_dir / snapshot_id1
        snapshot2_path = self.snapshot_dir / snapshot_id2

        # Get file lists
        files1 = {
            str(f.relative_to(snapshot1_path))
            for f in snapshot1_path.rglob("*")
            if f.is_file() and f.name != ".snapshot_metadata.json"
        }
        files2 = {
            str(f.relative_to(snapshot2_path))
            for f in snapshot2_path.rglob("*")
            if f.is_file() and f.name != ".snapshot_metadata.json"
        }

        # Find differences
        added = files2 - files1
        removed = files1 - files2
        common = files1 & files2

        # Check modified files
        modified = []
        for file_path in common:
            file1 = snapshot1_path / file_path
            file2 = snapshot2_path / file_path

            # Simple content comparison
            try:
                content1 = file1.read_text(encoding='utf-8')
                content2 = file2.read_text(encoding='utf-8')

                if content1 != content2:
                    modified.append(file_path)
            except Exception:
                # Binary files or other errors
                if file1.stat().st_size != file2.stat().st_size:
                    modified.append(file_path)

        diff = {
            "snapshot1": snapshot_id1,
            "snapshot2": snapshot_id2,
            "added": list(added),
            "removed": list(removed),
            "modified": modified,
            "total_changes": len(added) + len(removed) + len(modified)
        }

        # Add detailed file changes for markdown files
        detailed_changes = {}
        for file_path in modified:
            if file_path.endswith('.md'):
                file1 = snapshot1_path / file_path
                file2 = snapshot2_path / file_path

                try:
                    content1 = file1.read_text(encoding='utf-8')
                    content2 = file2.read_text(encoding='utf-8')

                    # Simple line-based diff
                    lines1 = content1.split('\n')
                    lines2 = content2.split('\n')

                    detailed_changes[file_path] = {
                        "lines_before": len(lines1),
                        "lines_after": len(lines2),
                        "size_before": len(content1),
                        "size_after": len(content2)
                    }
                except Exception as e:
                    LOG.warning(f"Could not analyze {file_path}: {e}")

        if detailed_changes:
            diff["detailed_changes"] = detailed_changes

        LOG.info(f"Diff between {snapshot_id1} and {snapshot_id2}: {diff['total_changes']} changes")
        return diff
