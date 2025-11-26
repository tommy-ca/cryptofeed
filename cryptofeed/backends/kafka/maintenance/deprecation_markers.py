"""
Deprecation marker management for documentation.

This module provides tools to insert, update, and manage deprecation markers
in documentation files, ensuring clear communication of deprecation timelines.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from packaging import version


LOG = logging.getLogger("feedhandler")


class DeprecationMarkerManager:
    """
    Manage deprecation markers in documentation files.

    Handles insertion, updates, and removal of deprecation warnings
    in markdown documentation.
    """

    def __init__(self):
        """Initialize the deprecation marker manager."""
        LOG.debug("DeprecationMarkerManager initialized")

    def insert_deprecation_marker(
        self,
        content: str,
        component: str,
        deprecation_info: Dict[str, str]
    ) -> str:
        """
        Insert a deprecation marker for a component in documentation.

        Args:
            content: Original documentation content
            component: Component name to mark as deprecated
            deprecation_info: Dict with 'version', 'removal', 'message' keys

        Returns:
            Updated documentation content with deprecation marker
        """
        deprecation_version = deprecation_info.get("version", "unknown")
        removal_version = deprecation_info.get("removal", "unknown")
        message = deprecation_info.get("message", "")

        # Create deprecation marker
        marker = (
            f"\n\n⚠️ **DEPRECATED** (since v{deprecation_version}, "
            f"will be removed in v{removal_version}): {message}\n"
        )

        # Find component section and insert marker after it
        # Look for lines like "- `component_name`" or "## ComponentName"
        pattern = rf"([-*]\s+`{component}`|##\s+{component})"
        match = re.search(pattern, content)

        if match:
            # Insert marker after the component line
            insert_pos = match.end()
            # Find end of the line
            newline_pos = content.find('\n', insert_pos)
            if newline_pos == -1:
                newline_pos = len(content)

            updated = content[:newline_pos] + marker + content[newline_pos:]
            LOG.info(f"Inserted deprecation marker for {component}")
            return updated
        else:
            # If we can't find the component, append at the end
            LOG.warning(f"Could not find {component} in content, appending marker")
            return content + marker

    def update_timeline(
        self,
        content: str,
        phase: str,
        new_date: str,
        reason: str
    ) -> str:
        """
        Update deprecation timeline when milestones change.

        Args:
            content: Original timeline documentation
            phase: Phase name to update (e.g., "Phase 2")
            new_date: New date for the phase
            reason: Reason for the timeline change

        Returns:
            Updated timeline documentation
        """
        # Find the phase section and update the date
        # Pattern: "## Phase X: Description (OLD_DATE)"
        pattern = rf"(##\s+{phase}[^(]+)\([^)]+\)"
        replacement = rf"\1({new_date})"

        updated = re.sub(pattern, replacement, content)

        if updated != content:
            LOG.info(f"Updated {phase} timeline to {new_date}: {reason}")
        else:
            LOG.warning(f"Could not find {phase} in timeline content")

        return updated

    def scan_for_markers(self, docs_dir: str) -> List[Dict[str, Any]]:
        """
        Scan all documentation files for existing deprecation markers.

        Args:
            docs_dir: Root directory of documentation files

        Returns:
            List of dictionaries with marker information
        """
        markers = []
        docs_path = Path(docs_dir)

        # Find all markdown files
        for md_file in docs_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')

                # Look for deprecation markers
                # Pattern: DEPRECATED (v1.0.0) or DEPRECATED (since v1.0.0, will be removed in v2.0.0)
                # Capture the entire line or sentence containing the deprecation
                pattern = r"[^\n]*?(DEPRECATED|deprecated)[^\n]*?v[\d.]+[^\n]*"

                for match in re.finditer(pattern, content, re.IGNORECASE):
                    markers.append({
                        "file": str(md_file),
                        "content": match.group(0).strip(),
                        "position": match.start()
                    })

            except Exception as e:
                LOG.error(f"Error scanning {md_file}: {e}")

        LOG.info(f"Found {len(markers)} deprecation markers in {docs_dir}")
        return markers

    def remove_expired_markers(
        self,
        content: str,
        current_version: str
    ) -> str:
        """
        Remove deprecation markers for components that have been removed.

        Args:
            content: Original documentation content
            current_version: Current version string (e.g., "2.5.0")

        Returns:
            Cleaned documentation content with expired markers removed
        """
        # Parse current version
        try:
            curr_ver = version.parse(current_version)
        except Exception as e:
            LOG.error(f"Invalid version format {current_version}: {e}")
            return content

        # Find sections with deprecation markers
        # Pattern: ## Component followed by any content including DEPRECATED marker
        # up to the next ## or end of document
        pattern = r"##\s+(\w+)\s*\n+\s*\*\*DEPRECATED[^*]*?removed in v([\d.]+)[^*]*?\*\*[^\n]*\n+.*?(?=\n##|\Z)"

        def check_expired(match):
            component_name = match.group(1)
            removal_version_str = match.group(2)

            try:
                removal_ver = version.parse(removal_version_str)
                if curr_ver >= removal_ver:
                    # Current version is past removal version, so remove this section
                    LOG.info(f"Removing expired deprecation section for {component_name}")
                    return ""
                else:
                    # Keep this section
                    return match.group(0)
            except Exception as e:
                LOG.warning(f"Could not parse version {removal_version_str}: {e}")
                # Keep the section if we can't parse
                return match.group(0)

        cleaned = re.sub(pattern, check_expired, content, flags=re.DOTALL)
        return cleaned
