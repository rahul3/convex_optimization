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
        from deblur_denoise.primal_dr.algorithm import primal_dr_splitting_test
        print("Running Primal Douglas-Rachford Splitting (Spingarn's Method) algorithm...")
        primal_dr_splitting_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                                blur_type='gaussian')
        print("Primal Douglas-Rachford Splitting (Spingarn's Method) algorithm completed successfully!")
    except Exception as e:
        print(f"Error running primal dr splitting test: {e}")
        sys.exit(1)

    try:
        # Testing primal dual dr splitting
        from deblur_denoise.primal_dual_dr.algorithm import test_primal_dual_dr_splitting
        print("Running Primal-Dual Douglas Rachford Splitting algorithm...")
        test_primal_dual_dr_splitting(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                                     blur_type='gaussian')
        print("Primal-Dual Douglas Rachford Splitting algorithm completed successfully!")
    except Exception as e:
        print(f"Error running primal dual dr splitting test: {e}")
        sys.exit(1)

    try:
        # Testing admm 
        from deblur_denoise.admm.algorithm import admm_solver_test
        print("Running ADMM algorithm...")
        admm_solver_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                         blur_type='gaussian')
        print("ADMM algorithm completed successfully!")
    except Exception as e:
        print(f"Error running admm solver test: {e}")
        sys.exit(1)

    try:
        # Testing chambolle pock
        from deblur_denoise.chambolle_pock.algorithm import chambolle_pock_test
        print("Running Chambolle-Pock algorithm...")
        chambolle_pock_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                            blur_type='gaussian')
        print("Chambolle-Pock algorithm completed successfully!")
    except Exception as e:
        print(f"Error running chambolle pock test: {e}")
        sys.exit(1)


if __name__ == "__main__":
    reinstall_package()
    run_tests() 