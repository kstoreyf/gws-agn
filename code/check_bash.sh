#!/usr/bin/env bash

echo "Bash version: $BASH_VERSION"
echo "Shell: $SHELL"
echo "Which bash: $(which bash)"

# Test process substitution
echo "Testing process substitution..."
if cat <(echo "test") 2>/dev/null; then
    echo "✓ Process substitution works"
else
    echo "✗ Process substitution does NOT work"
fi
