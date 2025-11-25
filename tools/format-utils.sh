#!/bin/bash
# Convenience script for scoped formatting commands in Cryptofeed
# Usage: ./tools/format-utils.sh <command> [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    echo "Cryptofeed Scoped Formatting Utilities"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  format-staged     Format only staged Python files"
    echo "  format-unstaged   Format only unstaged Python files"
    echo "  format-all        Format all changed Python files (staged + unstaged)"
    echo "  lint-staged       Lint only staged Python files"
    echo "  lint-unstaged     Lint only unstaged Python files"
    echo "  lint-all          Lint all changed Python files"
    echo "  dry-run-staged    Show what would be formatted (staged files)"
    echo "  dry-run-unstaged  Show what would be formatted (unstaged files)"
    echo "  dry-run-all       Show what would be formatted (all files)"
    echo "  status            Show current git status and changed files"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 format-staged"
    echo "  $0 dry-run-unstaged"
    echo "  $0 status"
}

# Function to get changed files
get_changed_files() {
    local change_type="$1"
    case "$change_type" in
        staged)
            git diff --cached --name-only -- '*.py'
            ;;
        unstaged)
            git diff --name-only -- '*.py'
            ;;
        all)
            git diff HEAD --name-only -- '*.py'
            ;;
        *)
            echo "Invalid change type: $change_type" >&2
            return 1
            ;;
    esac
}

# Function to run formatter
run_formatter() {
    local change_type="$1"
    local dry_run="$2"

    echo -e "${BLUE}🔍 Finding changed Python files ($change_type)...${NC}"

    local files
    files=$(get_changed_files "$change_type")

    if [ -z "$files" ]; then
        echo -e "${YELLOW}No changed Python files found for $change_type changes.${NC}"
        return 0
    fi

    echo -e "${GREEN}Found changed files:${NC}"
    echo "$files" | while read -r file; do
        [ -n "$file" ] && echo "  📄 $file"
    done

    if [ "$dry_run" = "true" ]; then
        echo -e "\n${YELLOW}DRY RUN - Would format with:${NC}"
        echo "  🛠️  ruff format"
        echo "  🛠️  isort"
        return 0
    fi

    echo -e "\n${BLUE}🚀 Running formatters...${NC}"

    # Run ruff format
    if echo "$files" | xargs ruff format; then
        echo -e "${GREEN}✅ ruff format completed${NC}"
    else
        echo -e "${RED}❌ ruff format failed${NC}"
        return 1
    fi

    # Run isort
    if echo "$files" | xargs isort --jobs 8; then
        echo -e "${GREEN}✅ isort completed${NC}"
    else
        echo -e "${RED}❌ isort failed${NC}"
        return 1
    fi

    echo -e "\n${GREEN}🎉 Formatting completed successfully!${NC}"
    echo -e "${YELLOW}💡 Remember to stage your changes: git add <files>${NC}"
}

# Function to run linter
run_linter() {
    local change_type="$1"

    echo -e "${BLUE}🔍 Finding changed Python files ($change_type)...${NC}"

    local files
    files=$(get_changed_files "$change_type")

    if [ -z "$files" ]; then
        echo -e "${YELLOW}No changed Python files found for $change_type changes.${NC}"
        return 0
    fi

    echo -e "${GREEN}Found changed files:${NC}"
    echo "$files" | while read -r file; do
        [ -n "$file" ] && echo "  📄 $file"
    done

    echo -e "\n${BLUE}🚀 Running ruff linter...${NC}"

    if echo "$files" | xargs ruff check --fix; then
        echo -e "${GREEN}✅ Linting completed successfully!${NC}"
    else
        echo -e "${RED}❌ Linting failed${NC}"
        return 1
    fi
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Current Git Status:${NC}"
    git status --short

    echo -e "\n${BLUE}🔍 Changed Python Files:${NC}"

    local staged_files unstaged_files

    staged_files=$(get_changed_files staged)
    unstaged_files=$(get_changed_files unstaged)

    if [ -n "$staged_files" ]; then
        echo -e "\n${GREEN}Staged:${NC}"
        echo "$staged_files" | while read -r file; do
            [ -n "$file" ] && echo "  ✅ $file"
        done
    fi

    if [ -n "$unstaged_files" ]; then
        echo -e "\n${YELLOW}Unstaged:${NC}"
        echo "$unstaged_files" | while read -r file; do
            [ -n "$file" ] && echo "  📝 $file"
        done
    fi

    if [ -z "$staged_files" ] && [ -z "$unstaged_files" ]; then
        echo -e "${GREEN}No changed Python files.${NC}"
    fi
}

# Main command handling
case "${1:-help}" in
    format-staged)
        run_formatter staged false
        ;;
    format-unstaged)
        run_formatter unstaged false
        ;;
    format-all)
        run_formatter all false
        ;;
    lint-staged)
        run_linter staged
        ;;
    lint-unstaged)
        run_linter unstaged
        ;;
    lint-all)
        run_linter all
        ;;
    dry-run-staged)
        run_formatter staged true
        ;;
    dry-run-unstaged)
        run_formatter unstaged true
        ;;
    dry-run-all)
        run_formatter all true
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}" >&2
        echo ""
        usage
        exit 1
        ;;
esac