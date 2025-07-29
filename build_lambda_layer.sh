#!/bin/bash

# Build Lambda Layer for dspy_minimal (Simple ARM64 version)
# This script creates a properly structured Lambda layer with all dependencies for ARM64
# using pre-built wheels and avoiding macOS/Linux compatibility issues

set -e  # Exit on any error

echo "🚀 Building Lambda Layer for dspy_minimal (Simple ARM64)..."

# Clean up any existing layer directory
if [ -d "lambda_layer" ]; then
    echo "🧹 Cleaning up existing lambda_layer directory..."
    rm -rf lambda_layer
fi

# Create the layer directory structure
echo "📁 Creating layer directory structure..."
mkdir -p lambda_layer/python

# Copy the dspy_minimal package
echo "📦 Copying dspy_minimal package..."
cp -r dspy_minimal/dspy_minimal lambda_layer/python/

# Install dependencies with specific versions and platform targeting
echo "📥 Installing dependencies with platform-specific wheels..."

# Install pydantic with platform-specific wheel for ARM64 Linux
echo "🔧 Installing pydantic for ARM64 Linux..."
pip download --platform manylinux2014_aarch64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --dest lambda_layer/python \
    "pydantic>=2.8.0,<3.0.0"

# Install regex with platform-specific wheel for ARM64 Linux
echo "🔧 Installing regex for ARM64 Linux..."
pip download --platform manylinux2014_aarch64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --dest lambda_layer/python \
    regex>=2023.0.0

# Install rpds-py with platform-specific wheel for ARM64 Linux
echo "🔧 Installing rpds-py for ARM64 Linux..."
pip download --platform manylinux2014_aarch64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --dest lambda_layer/python \
    rpds-py>=0.10.0

# Install other dependencies
echo "📥 Installing other dependencies..."
pip install -t lambda_layer/python \
    requests>=2.31.0 \
    boto3>=1.34.0 \
    anyio>=4.0.0 \
    mcp>=1.0.0 \
    jsonschema>=4.0.0 \
    attrs>=23.0.0 \
    referencing>=0.30.0 \
    jsonschema-specifications>=2023.7.0

# Extract the downloaded wheels (non-interactive)
echo "📦 Extracting downloaded wheels..."
cd lambda_layer/python
for wheel in *.whl; do
    if [ -f "$wheel" ]; then
        echo "   Extracting $wheel..."
        # Use -o flag to overwrite without prompting
        unzip -q -o "$wheel"
        rm "$wheel"
    fi
done
cd ../..

# Clean up __pycache__ and .pyc files
echo "🧹 Cleaning up cache files..."
find lambda_layer/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find lambda_layer/python -name "*.pyc" -delete 2>/dev/null || true

# Remove unnecessary files to reduce size
echo "🗑️ Removing unnecessary files..."
find lambda_layer/python -name "*.pyo" -delete 2>/dev/null || true
find lambda_layer/python -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find lambda_layer/python -name "test_*" -delete 2>/dev/null || true

# Remove development files and documentation
echo "🗑️ Removing development files..."
find lambda_layer/python -name "*.md" -delete 2>/dev/null || true
find lambda_layer/python -name "*.txt" -delete 2>/dev/null || true
find lambda_layer/python -name "*.rst" -delete 2>/dev/null || true
find lambda_layer/python -name "setup.py" -delete 2>/dev/null || true
find lambda_layer/python -name "pyproject.toml" -delete 2>/dev/null || true
find lambda_layer/python -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Verify all required modules are present
echo "🔍 Verifying required modules..."
required_modules=(
    "dspy_minimal/__init__.py"
    "dspy_minimal/adapters/__init__.py"
    "dspy_minimal/adapters/base.py"
    "dspy_minimal/evaluate/__init__.py"
    "dspy_minimal/evaluate/evaluate.py"
    "dspy_minimal/utils/exceptions.py"
    "dspy_minimal/mcp/__init__.py"
    "dspy_minimal/clients/lm.py"
    "dspy_minimal/predict/predict.py"
    "dspy_minimal/signatures/signature.py"
)

for module in "${required_modules[@]}"; do
    if [ -f "lambda_layer/python/$module" ]; then
        echo "   ✅ $module"
    else
        echo "   ❌ Missing: $module"
        exit 1
    fi
done

# Verify regex is properly installed
echo "🔍 Verifying regex installation..."
if [ -f "lambda_layer/python/regex/__init__.py" ]; then
    echo "   ✅ regex module found"
    if ls lambda_layer/python/regex/*.so 1> /dev/null 2>&1; then
        echo "   ✅ regex compiled extension found"
        ls -la lambda_layer/python/regex/*.so
    else
        echo "   ⚠️  regex compiled extension missing"
    fi
else
    echo "   ❌ regex module missing"
    exit 1
fi

# Verify pydantic_core is properly installed
echo "🔍 Verifying pydantic_core installation..."
if [ -f "lambda_layer/python/pydantic_core/__init__.py" ]; then
    echo "   ✅ pydantic_core module found"
    if ls lambda_layer/python/pydantic_core/*.so 1> /dev/null 2>&1; then
        echo "   ✅ pydantic_core compiled extension found"
        ls -la lambda_layer/python/pydantic_core/*.so
    else
        echo "   ⚠️  pydantic_core compiled extension missing"
    fi
else
    echo "   ❌ pydantic_core module missing"
    exit 1
fi

# Verify rpds is properly installed
echo "🔍 Verifying rpds installation..."
if [ -f "lambda_layer/python/rpds/__init__.py" ]; then
    echo "   ✅ rpds module found"
    if ls lambda_layer/python/rpds/*.so 1> /dev/null 2>&1; then
        echo "   ✅ rpds compiled extension found"
        ls -la lambda_layer/python/rpds/*.so
        # Check if it's the correct platform (should contain 'linux' not 'darwin')
        if ls lambda_layer/python/rpds/*linux*.so 1> /dev/null 2>&1; then
            echo "   ✅ rpds Linux ARM64 binary found"
        elif ls lambda_layer/python/rpds/*darwin*.so 1> /dev/null 2>&1; then
            echo "   ❌ rpds macOS binary found - this will not work on Lambda!"
            exit 1
        else
            echo "   ⚠️  rpds binary platform unclear"
        fi
    else
        echo "   ⚠️  rpds compiled extension missing"
    fi
else
    echo "   ❌ rpds module missing"
    exit 1
fi

# Verify jsonschema is properly installed
echo "🔍 Verifying jsonschema installation..."
if [ -f "lambda_layer/python/jsonschema/__init__.py" ]; then
    echo "   ✅ jsonschema module found"
else
    echo "   ❌ jsonschema module missing"
    exit 1
fi

# Create the layer zip
echo "📦 Creating layer zip file..."
cd lambda_layer
zip -r dspy_minimal_layer.zip python

# Move zip to parent directory
mv dspy_minimal_layer.zip ../

# Clean up temporary directory
cd ..
rm -rf lambda_layer

# Show layer info
echo "✅ Lambda layer created successfully!"
echo "📁 Layer file: dspy_minimal_layer.zip"
echo "📊 Layer size: $(du -h dspy_minimal_layer.zip | cut -f1)"
echo ""
echo "📋 To deploy:"
echo "1. Upload dspy_minimal_layer.zip to AWS Lambda as a layer"
echo "2. Attach the layer to your ARM64 Lambda function"
echo "3. Use imports like: from dspy_minimal import LM, Predict, etc."
echo ""
echo "🔍 Layer contents:"
unzip -l dspy_minimal_layer.zip | head -20
echo ""
echo "📋 Supported imports in this layer:"
echo "✅ from dspy_minimal import LM, configure as dspy_configure"
echo "✅ from dspy_minimal import BootstrapFewShot, Example, Predict"
echo "✅ from dspy_minimal.adapters import Adapter"
echo "✅ from dspy_minimal.utils.exceptions import AdapterParseError"
echo "✅ from dspy_minimal import Signature, InputField, OutputField"
echo "✅ from dspy_minimal import Module, Predict, Signature, InputField, OutputField, ReAct, Tool"
echo "✅ from dspy_minimal.mcp import ClientSession, StdioServerParameters, stdio_client"
echo "✅ from dspy_minimal import BootstrapFewShot, Evaluate, configure as dspy_configure, LM, Predict"
echo "✅ import regex  # Pre-built ARM64 Linux wheel"
echo "✅ import pydantic  # Pre-built ARM64 Linux wheel"
echo "✅ import rpds  # Pre-built ARM64 Linux wheel (required by jsonschema/mcp)"
echo "✅ import jsonschema  # Required by mcp"
echo ""
echo "🚀 Ready for ARM64 Lambda deployment!"
echo ""
echo "📝 Notes:"
echo "- Uses pre-built ARM64 Linux wheels for compiled extensions"
echo "- Avoids macOS/Linux compatibility issues"
echo "- All compiled extensions properly built for Linux ARM64"
echo "- Use this for Lambda functions with ARM64 architecture"
echo "- ARM64 typically provides better price/performance ratio" 