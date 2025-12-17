#!/usr/bin/env bash
# Docker image build script with semantic versioning and metadata labels
#
# This script:
# - Extracts Git metadata (version, commit SHA, branch)
# - Generates semantic version tags (latest, vX.Y.Z, vX.Y, vX)
# - Tags images with Git commit SHA for traceability
# - Applies metadata labels (version, build_timestamp, git_commit_sha, git_branch)
# - Supports multiple tag application simultaneously
#
# Usage:
#   ./build.sh [IMAGE_NAME] [REGISTRY]
#
# Examples:
#   ./build.sh                        # Build as cryptofeed:latest locally
#   ./build.sh myapp docker.io/org    # Build as docker.io/org/myapp:latest
#
# Environment Variables:
#   IMAGE_NAME - Override image name (default: cryptofeed)
#   REGISTRY   - Container registry prefix (default: none)
#   PUSH       - Push to registry if set to "true" (default: false)

set -euo pipefail

# Configuration
IMAGE_NAME="${1:-${IMAGE_NAME:-cryptofeed}}"
REGISTRY="${2:-${REGISTRY:-}}"
PUSH="${PUSH:-false}"

# Add registry prefix if provided
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}"
fi

echo "====================================================================="
echo "Building Docker image: ${FULL_IMAGE_NAME}"
echo "====================================================================="

# Extract Git metadata
echo "Extracting Git metadata..."

# Get version from git describe (e.g., v3.0.0-446-g7d74c6d5)
GIT_DESCRIBE=$(git describe --tags --always 2>/dev/null || echo "dev")
echo "  Git describe: ${GIT_DESCRIBE}"

# Extract semantic version if on tagged commit (e.g., v1.2.3)
if [[ $GIT_DESCRIBE =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    VERSION="$GIT_DESCRIBE"
    echo "  Semantic version detected: ${VERSION}"
else
    # Use git describe output as version (includes commit count and short SHA)
    VERSION="$GIT_DESCRIBE"
    echo "  Development version: ${VERSION}"
fi

# Get full commit SHA (40 characters)
GIT_COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "  Git commit SHA: ${GIT_COMMIT_SHA}"

# Get short commit SHA (12 characters)
GIT_COMMIT_SHORT="${GIT_COMMIT_SHA:0:12}"
echo "  Git commit SHA (short): ${GIT_COMMIT_SHORT}"

# Get current branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "  Git branch: ${GIT_BRANCH}"

# Generate build timestamp in ISO 8601 format
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "  Build timestamp: ${BUILD_TIMESTAMP}"

echo ""
echo "Generating image tags..."

# Initialize tags array with latest
TAGS="-t ${FULL_IMAGE_NAME}:latest"
echo "  ${FULL_IMAGE_NAME}:latest"

# Add semantic version tags if on tagged commit
if [[ $VERSION =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    MAJOR="${BASH_REMATCH[1]}"
    MINOR="${BASH_REMATCH[2]}"
    PATCH="${BASH_REMATCH[3]}"

    # Full semantic version (vX.Y.Z)
    TAGS="$TAGS -t ${FULL_IMAGE_NAME}:v${MAJOR}.${MINOR}.${PATCH}"
    echo "  ${FULL_IMAGE_NAME}:v${MAJOR}.${MINOR}.${PATCH}"

    # Minor version tag (vX.Y)
    TAGS="$TAGS -t ${FULL_IMAGE_NAME}:v${MAJOR}.${MINOR}"
    echo "  ${FULL_IMAGE_NAME}:v${MAJOR}.${MINOR}"

    # Major version tag (vX)
    TAGS="$TAGS -t ${FULL_IMAGE_NAME}:v${MAJOR}"
    echo "  ${FULL_IMAGE_NAME}:v${MAJOR}"
fi

# Add commit SHA tag for traceability
COMMIT_TAG="commit-${GIT_COMMIT_SHORT}"
TAGS="$TAGS -t ${FULL_IMAGE_NAME}:${COMMIT_TAG}"
echo "  ${FULL_IMAGE_NAME}:${COMMIT_TAG}"

echo ""
echo "Building image with metadata labels..."
echo "  ARG VERSION=${VERSION}"
echo "  ARG BUILD_TIMESTAMP=${BUILD_TIMESTAMP}"
echo "  ARG GIT_COMMIT_SHA=${GIT_COMMIT_SHA}"
echo "  ARG GIT_BRANCH=${GIT_BRANCH}"
echo ""

# Build image with all tags and metadata labels
docker build \
    $TAGS \
    --build-arg VERSION="${VERSION}" \
    --build-arg BUILD_TIMESTAMP="${BUILD_TIMESTAMP}" \
    --build-arg GIT_COMMIT_SHA="${GIT_COMMIT_SHA}" \
    --build-arg GIT_BRANCH="${GIT_BRANCH}" \
    .

echo ""
echo "====================================================================="
echo "Build complete!"
echo "====================================================================="
echo ""
echo "Image tags created:"
docker images "${FULL_IMAGE_NAME}" --format "  {{.Repository}}:{{.Tag}}" | head -10

# Inspect image labels
echo ""
echo "Image metadata labels:"
docker inspect "${FULL_IMAGE_NAME}:latest" \
    --format '  version: {{index .Config.Labels "version"}}
  build_timestamp: {{index .Config.Labels "build_timestamp"}}
  git_commit_sha: {{index .Config.Labels "git_commit_sha"}}
  git_branch: {{index .Config.Labels "git_branch"}}'

# Push to registry if requested
if [ "$PUSH" = "true" ]; then
    echo ""
    echo "Pushing images to registry..."

    # Push all tags
    for tag in $(docker images "${FULL_IMAGE_NAME}" --format "{{.Tag}}" | head -10); do
        echo "  Pushing ${FULL_IMAGE_NAME}:${tag}..."
        docker push "${FULL_IMAGE_NAME}:${tag}"
    done

    echo ""
    echo "Push complete!"
fi

echo ""
echo "====================================================================="
echo "Done!"
echo "====================================================================="
