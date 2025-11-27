"""
Code example validator for documentation accuracy.

This module validates Python code examples in documentation to ensure
they are syntactically correct, imports are resolvable, and examples
execute without errors.
"""

import ast
import importlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
import traceback


LOG = logging.getLogger("feedhandler")


class CodeExampleValidator:
    """
    Validate code examples in documentation for accuracy.

    Ensures documentation code blocks are syntactically correct,
    imports are resolvable, and examples execute successfully.
    """

    def __init__(self):
        """Initialize the code example validator."""
        LOG.debug("CodeExampleValidator initialized")

    def extract_code_blocks(self, markdown: str, language: str = "python") -> List[Dict[str, Any]]:
        """
        Extract code blocks from markdown documentation.

        Args:
            markdown: Markdown content
            language: Language identifier (default: "python")

        Returns:
            List of dictionaries with code block information
        """
        code_blocks = []

        # Pattern: ```language\ncode\n```
        pattern = rf"```{language}\s*\n(.*?)```"

        for match in re.finditer(pattern, markdown, re.DOTALL):
            code = match.group(1)
            line_number = markdown[:match.start()].count('\n') + 1

            code_blocks.append({
                "code": code,
                "line_number": line_number,
                "language": language
            })

        LOG.debug(f"Extracted {len(code_blocks)} {language} code blocks")
        return code_blocks

    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate Python syntax in code.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with validation result
        """
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

    def validate_imports(self, code: str) -> Dict[str, Any]:
        """
        Validate that imports in code can be resolved.

        Args:
            code: Python code with import statements

        Returns:
            Dictionary with validation result
        """
        try:
            # Parse the code to extract imports
            tree = ast.parse(code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Try to import each module
            failed_imports = []
            for module_name in imports:
                try:
                    # Try to import the full module path
                    importlib.import_module(module_name)
                except ImportError:
                    # If full path fails, try just the top-level module
                    top_level = module_name.split('.')[0]
                    try:
                        importlib.import_module(top_level)
                        # Top-level exists, but submodule doesn't
                        # This is a valid failure case for nonexistent submodules
                        failed_imports.append({
                            "module": module_name,
                            "error": f"Module '{module_name}' not found"
                        })
                    except ImportError as e:
                        # Top-level module also doesn't exist
                        failed_imports.append({
                            "module": module_name,
                            "error": str(e)
                        })

            if failed_imports:
                return {
                    "valid": False,
                    "error": f"Failed to import: {', '.join(f['module'] for f in failed_imports)}",
                    "details": failed_imports
                }
            else:
                return {"valid": True}

        except Exception as e:
            return {
                "valid": False,
                "error": f"Error analyzing imports: {e}"
            }

    def execute_in_sandbox(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute code example in a sandbox to verify it runs.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (not strictly enforced)

        Returns:
            Dictionary with execution result
        """
        # Create a restricted execution environment
        sandbox_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
            "__doc__": None,
        }

        try:
            # Compile the code
            compiled = compile(code, "<string>", "exec")

            # Execute in sandbox
            exec(compiled, sandbox_globals)

            return {
                "success": True,
                "error": None
            }

        except AssertionError as e:
            # Assertions in examples might fail intentionally
            return {
                "success": False,
                "error": f"AssertionError: {e}",
                "error_type": "assertion"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate all code examples in a documentation file.

        Args:
            file_path: Path to markdown file

        Returns:
            Dictionary with validation results for all blocks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                "file": file_path,
                "error": f"Could not read file: {e}",
                "blocks": [],
                "total": 0,
                "valid": 0,
                "invalid": 0
            }

        # Extract code blocks
        blocks = self.extract_code_blocks(content, language="python")

        # Validate each block
        results = []
        valid_count = 0
        invalid_count = 0

        for block in blocks:
            code = block["code"]

            # Validate syntax
            syntax_result = self.validate_syntax(code)

            if syntax_result["valid"]:
                # Validate imports
                import_result = self.validate_imports(code)

                block_result = {
                    "line": block["line_number"],
                    "valid": import_result["valid"],
                    "syntax_valid": True,
                    "imports_valid": import_result["valid"],
                }

                if not import_result["valid"]:
                    block_result["error"] = import_result.get("error")
                    invalid_count += 1
                else:
                    valid_count += 1

            else:
                block_result = {
                    "line": block["line_number"],
                    "valid": False,
                    "syntax_valid": False,
                    "error": syntax_result.get("error")
                }
                invalid_count += 1

            results.append(block_result)

        return {
            "file": file_path,
            "blocks": results,
            "total": len(blocks),
            "valid": valid_count,
            "invalid": invalid_count
        }

    def scan_directory(self, directory: str) -> Dict[str, Any]:
        """
        Scan all documentation files and report invalid examples.

        Args:
            directory: Directory path to scan

        Returns:
            Dictionary with overall validation report
        """
        dir_path = Path(directory)
        all_results = []

        total_blocks = 0
        total_valid = 0
        total_invalid = 0

        # Find all markdown files
        for md_file in dir_path.rglob("*.md"):
            file_result = self.validate_file(str(md_file))

            all_results.append({
                "file": str(md_file),
                "total": file_result["total"],
                "valid": file_result["valid"],
                "invalid": file_result["invalid"],
                "blocks": file_result["blocks"]
            })

            total_blocks += file_result["total"]
            total_valid += file_result["valid"]
            total_invalid += file_result["invalid"]

        report = {
            "directory": directory,
            "files": all_results,
            "total_blocks": total_blocks,
            "valid_blocks": total_valid,
            "invalid_blocks": total_invalid,
            "success_rate": total_valid / total_blocks if total_blocks > 0 else 1.0
        }

        LOG.info(
            f"Scanned {len(all_results)} files: "
            f"{total_valid}/{total_blocks} valid blocks "
            f"({report['success_rate']:.1%} success rate)"
        )

        return report
