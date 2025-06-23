#!/usr/bin/env python3
"""
Legacy demo script - redirects to new package structure.
For the new demo, use: python examples/demo.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🔄 Redirecting to new package structure...")
    print("   This script now uses the BioFlux package.")
    print("   For the enhanced demo, use: python examples/demo.py")
    print()
    
    # Run the new demo
    examples_dir = Path(__file__).parent / "examples"
    demo_script = examples_dir / "demo.py"
    
    if demo_script.exists():
        return subprocess.call([sys.executable, str(demo_script)])
    else:
        print("❌ Demo script not found.")
        print("   Please run: python examples/demo.py")
        return 1

if __name__ == "__main__":
    exit(main())
