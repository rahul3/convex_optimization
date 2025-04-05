import subprocess
import sys
import os

def reinstall_package():
    """Uninstall and reinstall the package in development mode"""
    print("Reinstalling package...")
    try:
        # Uninstall the package
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "convex_optimization", "-y"], check=True)
        # Install in development mode
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("Package reinstalled successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error reinstalling package: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite"""
    print("\nRunning tests...")
    try:
        # Import and run the test function

        # Testing admm
        from deblur_denoise.admm.algorithm import admm_solver_test
        admm_solver_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg')

        # Testing chambolle pock
        from deblur_denoise.chambolle_pock.algorithm import chambolle_pock_test
        chambolle_pock_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg')
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reinstall_package()
    run_tests() 