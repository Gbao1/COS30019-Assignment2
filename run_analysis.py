"""
Quick setup and analysis runner for the traffic flow prediction project.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'pytest'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")

    # Check for TensorFlow separately
    try:
        import tensorflow
        print(f"✓ tensorflow (version: {tensorflow.__version__})")
    except ImportError:
        print("✗ tensorflow - Neural network models will be skipped")
        print("  To install TensorFlow: pip install tensorflow")

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    return True

def run_tests():
    """Run the test suite to verify everything works."""
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    try:
        # Add src to Python path
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Run tests
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "src/tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("✗ Some tests failed")
            return False

    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_model_comparison():
    """Run the main model comparison."""
    print("\n" + "="*60)
    print("RUNNING MODEL COMPARISON")
    print("="*60)

    try:
        # Run the main comparison script
        result = subprocess.run([
            sys.executable, "model_comparison.py"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✓ Model comparison completed successfully!")
            return True
        else:
            print("✗ Model comparison failed")
            return False

    except Exception as e:
        print(f"Error running model comparison: {e}")
        return False

def generate_visualizations():
    """Generate visualization plots."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    try:
        # Add src to Python path
        src_path = str(Path(__file__).parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Run visualization script
        result = subprocess.run([
            sys.executable, "-c",
            "from src.utils.visualization import main; main()"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✓ Visualizations generated successfully!")
            return True
        else:
            print("✗ Visualization generation failed")
            return False

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False

def print_results_summary():
    """Print a summary of generated results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    results_dir = Path("results")
    if results_dir.exists():
        print("Generated files:")
        for file in results_dir.rglob("*"):
            if file.is_file():
                print(f"  - {file}")
    else:
        print("No results directory found")

    print(f"\nNext steps:")
    print(f"1. Check the REPORT.md file for the draft report")
    print(f"2. Review results in the 'results/' directory")
    print(f"3. Update the report with actual experimental results")
    print(f"4. Add your analysis and conclusions")

def main():
    """Main function to run the complete analysis pipeline."""
    print("Traffic Flow Prediction Model Comparison")
    print("="*60)
    print("Setting up and running complete analysis...")

    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies before continuing.")
        return

    # Run tests (optional, but recommended)
    print("\nWould you like to run tests first? (y/n): ", end="")
    run_tests_choice = input().lower().strip()
    if run_tests_choice == 'y':
        if not run_tests():
            print("\nSome tests failed. Continue anyway? (y/n): ", end="")
            continue_choice = input().lower().strip()
            if continue_choice != 'y':
                return

    # Run model comparison
    if not run_model_comparison():
        print("\nModel comparison failed. Please check the error messages above.")
        return

    # Generate visualizations
    generate_visualizations()

    # Print summary
    print_results_summary()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()