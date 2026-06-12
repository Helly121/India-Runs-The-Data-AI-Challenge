#!/usr/bin/env python3
"""
Entrypoint redirection script.
Allows running the ranking system from the root workspace directory.
"""

import sys
import os
import subprocess

def main():
    # Construct path to the main ranking system script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "submission", "ranking_system.py")
    
    # Run the main script with all passed arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]
    
    # Run process and wait
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
