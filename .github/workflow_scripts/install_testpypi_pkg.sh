#!/bin/bash  

# The $1 argument is the version number passed from the workflow  
VERSION=$1

echo "version: $VERSION"

for i in {1..5}; do  
  if python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dspy-ai-test=="$VERSION"; then  
    break  
  else  
    echo "Attempt $i failed. Waiting before retrying..."  
    sleep 10  
  fi  
done