"""
Automated documentation updater for component changes.

This module provides tools to automatically update documentation when
Kafka backend components change, ensuring documentation stays synchronized
with code.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


LOG = logging.getLogger("feedhandler")


class DocumentationAutoUpdater:
    """
    Automated documentation update system for component changes.

    Detects component changes and updates relevant documentation files
    to maintain synchronization between code and docs.
    """

    def __init__(self, docs_root: Optional[Path] = None):
        """
        Initialize the documentation auto-updater.

        Args:
            docs_root: Root directory of documentation files (defaults to docs/kafka/)
        """
        if docs_root is None:
            # Default to project docs/kafka/ directory
            import cryptofeed
            cryptofeed_root = Path(cryptofeed.__file__).parent.parent
            docs_root = cryptofeed_root / "docs" / "kafka"

        self.docs_root = Path(docs_root)
        LOG.debug(f"DocumentationAutoUpdater initialized with docs_root: {self.docs_root}")

    def detect_affected_docs(self, changes: Dict[str, Any]) -> List[str]:
        """
        Identify documentation files affected by component changes.

        Args:
            changes: Dictionary describing component changes

        Returns:
            List of affected documentation file paths
        """
        affected = []
        component = changes.get("component", "")

        # Mapping of components to affected documentation files
        component_doc_mapping = {
            "KafkaConfig": ["API_REFERENCE.md", "user-guide.md", "BEST_PRACTICES.md"],
            "KafkaTopicConfig": ["API_REFERENCE.md", "technical-specification.md"],
            "KafkaPartitionConfig": ["API_REFERENCE.md", "technical-specification.md"],
            "KafkaCallback": ["API_REFERENCE.md", "user-guide.md", "migration-guide.md"],
        }

        # Get affected docs for this component
        doc_patterns = component_doc_mapping.get(component, [])

        # Find actual files in docs directory
        for pattern in doc_patterns:
            matching_files = list(self.docs_root.rglob(pattern))
            affected.extend(str(f) for f in matching_files)

        LOG.info(f"Component {component} affects {len(affected)} documentation files")
        return affected

    def generate_field_documentation(self, component_info: Dict[str, Any]) -> str:
        """
        Generate documentation for new fields added to a component.

        Args:
            component_info: Component information including new_fields list

        Returns:
            Markdown documentation for the new fields
        """
        component_name = component_info.get("name", "Unknown")
        new_fields = component_info.get("new_fields", [])

        if not new_fields:
            return ""

        doc_lines = [f"\n### New Fields in {component_name}\n"]

        for field in new_fields:
            field_name = field.get("name", "unknown")
            field_type = field.get("type", "Any")
            default = field.get("default", "None")
            description = field.get("description", "No description available")

            doc_lines.append(f"- **`{field_name}`** (`{field_type}`, default: `{default}`): {description}")

        return "\n".join(doc_lines)

    def generate_deprecation_documentation(self, component_info: Dict[str, Any]) -> str:
        """
        Generate documentation for deprecated fields.

        Args:
            component_info: Component information including deprecated_fields list

        Returns:
            Markdown documentation for deprecated fields
        """
        component_name = component_info.get("name", "Unknown")
        deprecated_fields = component_info.get("deprecated_fields", [])

        if not deprecated_fields:
            return ""

        doc_lines = [f"\n### Deprecated Fields in {component_name}\n"]

        for field in deprecated_fields:
            field_name = field.get("name", "unknown")
            deprecated_version = field.get("deprecated_version", "unknown")
            removal_version = field.get("removal_version", "unknown")
            replacement = field.get("replacement", "No replacement specified")

            doc_lines.append(
                f"- **`{field_name}`** ⚠️ DEPRECATED (since v{deprecated_version}, "
                f"will be removed in v{removal_version}): {replacement}"
            )

        return "\n".join(doc_lines)

    def scan_for_outdated_references(self, file_path: str, deprecated_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Scan a documentation file for references to deprecated fields.

        Args:
            file_path: Path to documentation file
            deprecated_fields: List of deprecated field names to search for

        Returns:
            List of dictionaries with outdated reference information
        """
        outdated = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                for field in deprecated_fields:
                    # Look for field name in various contexts
                    if field in line:
                        # Check if it's in a code block or parameter reference
                        if '```' in line or field in re.findall(r'\b\w+\b', line):
                            outdated.append({
                                "field": field,
                                "file": file_path,
                                "line": line_num,
                                "content": line.strip()
                            })
                            break  # Only report once per line

        except Exception as e:
            LOG.error(f"Error scanning file {file_path}: {e}")

        return outdated
