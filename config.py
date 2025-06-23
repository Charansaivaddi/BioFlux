#!/usr/bin/env python3
"""
Legacy configuration script - redirects to new package structure.
For the new configuration testing, use: python examples/config_test.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🔄 Redirecting to new package structure...")
    print("   This script now uses the BioFlux package.")
    print("   For direct configuration testing, use: python examples/config_test.py")
    print()
    
    # Run the new configuration test
    examples_dir = Path(__file__).parent / "examples"
    config_script = examples_dir / "config_test.py"
    
    if config_script.exists():
        return subprocess.call([sys.executable, str(config_script)])
    else:
        print("❌ Configuration test script not found.")
        print("   Please run: python examples/config_test.py")
        return 1

if __name__ == "__main__":
    exit(main())
