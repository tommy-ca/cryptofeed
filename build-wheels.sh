#!/bin/bash
set -e -u -x

# Support Python 3.9-3.12 as per project requirements
py_vers=(
    "/opt/python/cp39-cp39/bin"
    "/opt/python/cp310-cp310/bin" 
    "/opt/python/cp311-cp311/bin"
    "/opt/python/cp312-cp312/bin"
)

# Use more compatible manylinux tag (manylinux_2_28 is widely supported)
PLAT=manylinux_2_28_x86_64

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Install uv for each Python version (uv is faster and more reliable than pip)
for PYBIN in "${py_vers[@]}"; do
    echo "Installing uv for ${PYBIN}"
    "${PYBIN}/python" -m pip install -U uv
done

# Install build dependencies using uv for each Python version
for PYBIN in "${py_vers[@]}"; do
    echo "Installing dependencies with uv for ${PYBIN}"
    "${PYBIN}/uv" pip install --system cython>=3.0.0 build setuptools wheel
done

# Build wheels using modern build backend with uv environment
for PYBIN in "${py_vers[@]}"; do
    echo "Building wheel with ${PYBIN} using uv"
    "${PYBIN}/python" -m build --wheel /io/ --outdir wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done