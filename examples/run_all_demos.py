#!/usr/bin/env python3
"""
All Models Demo Runner
======================

A comprehensive script that runs all available model demonstrations
and provides a complete overview of the BioFlux RL ecosystem.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def run_demo_script(script_name, description):
    """Run a demo script and handle errors gracefully."""
    print(f"\\n🚀 Running {description}")
    print("=" * (len(description) + 10))
    
    try:
        # Get the correct Python path
        python_path = "/Users/charan/.pyenv/versions/3.8.11/bin/python"
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        # Run the script
        result = subprocess.run([python_path, str(script_path)], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print("📋 Output:")
            # Print last few lines of output
            lines = result.stdout.strip().split('\\n')[-10:]
            for line in lines:
                print(f"   {line}")
            return True
        else:
            print(f"❌ {description} failed!")
            print("Error output:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Run all model demonstrations."""
    print("🎭 BioFlux Complete Model Demonstration Suite")
    print("=" * 55)
    print("Running all available model demonstrations and analyses...")
    print()
    
    # Track results
    results = {}
    
    # 1. Simple Showcase
    results['simple_showcase'] = run_demo_script(
        'simple_showcase.py', 
        'Simple Model Showcase'
    )
    time.sleep(2)
    
    # 2. Comprehensive Model Demos
    results['model_demos'] = run_demo_script(
        'model_demos.py',
        'Comprehensive Model Demonstrations'
    )
    time.sleep(2)
    
    # 3. Behavioral Analysis
    results['behavior_analysis'] = run_demo_script(
        'behavior_analysis.py',
        'Behavioral Analysis'
    )
    time.sleep(2)
    
    # 4. Inference Testing
    results['inference'] = run_demo_script(
        'inference.py',
        'Model Inference Testing'
    )
    time.sleep(2)
    
    # 5. Deployment Demo
    results['deployment'] = run_demo_script(
        'deployment.py',
        'Deployment Demonstration'
    )
    time.sleep(2)
    
    # 6. Real-Time Simulation Demo
    results['simulation_demo'] = run_demo_script(
        'simulation_demo.py',
        'Real-Time Simulation Demo'
    )
    
    # Final Summary
    print("\\n" + "=" * 55)
    print("🏆 DEMONSTRATION SUITE RESULTS")
    print("=" * 55)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"📊 Overall Success Rate: {successful}/{total} ({(successful/total)*100:.1f}%)")
    print()
    
    for demo, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {demo.replace('_', ' ').title()}: {status}")
    
    print()
    print("📁 Generated Files Location: output/")
    print("📋 Key Outputs:")
    
    # List key output files
    output_dir = Path(__file__).parent.parent / "output"
    if output_dir.exists():
        key_files = [
            "model_comparison_summary.png",
            "simple_model_showcase.png", 
            "training_results.png",
            "inference_results.json",
            "*_behavior_analysis.png",
            "*_simulation_demo.png",
            "*_simulation_data.json"
        ]
        
        for pattern in key_files:
            if "*" in pattern:
                matching_files = list(output_dir.glob(pattern))
                if matching_files:
                    print(f"   • {len(matching_files)} files matching {pattern}")
            else:
                file_path = output_dir / pattern
                if file_path.exists():
                    print(f"   • {pattern}")
    
    print()
    
    if successful == total:
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("🚀 Your BioFlux models are fully validated and ready for use!")
    elif successful > 0:
        print("⚠️  Some demonstrations completed successfully.")
        print("🔧 Check individual outputs for any issues that need attention.")
    else:
        print("❌ No demonstrations completed successfully.")
        print("🔧 Please check the installation and model files.")
    
    print()
    print("📖 For detailed results, see:")
    print("   • MODEL_DEMONSTRATIONS_SUMMARY.md")
    print("   • Individual visualization files in output/")
    print("   • Training and analysis JSON reports")
    
    print("\\n" + "=" * 55)
    print("🎭 BioFlux Model Demonstration Suite Complete!")
    print("=" * 55)

if __name__ == "__main__":
    main()
