import subprocess
import sys
import os
from deblur_denoise.utils.logging_utils import logger, log_execution_time


IMAGE_PATH = os.getenv('CO_IMAGE_PATH', '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg')

@log_execution_time(logger)
def reinstall_package():
    """Uninstall and reinstall the package in development mode"""
    logger.info("Reinstalling package...")
    try:
        # Uninstall the package
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "convex_optimization", "-y"], check=True)
        # Install in development mode
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        logger.info("Package reinstalled successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error reinstalling package: {e}")
        sys.exit(1)

@log_execution_time(logger)
def run_tests():
    """Run the test suite"""
    logger.info("\nRunning tests...")
    try:
        # Import and run the test function
        from deblur_denoise.primal_dr.algorithm import primal_dr_splitting_test
        logger.info("Running Primal Douglas-Rachford Splitting (Spingarn's Method) algorithm...")
        primal_dr_splitting_test(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                                blur_type='gaussian')
        logger.info("Primal Douglas-Rachford Splitting (Spingarn's Method) algorithm completed successfully!")
    except Exception as e:
        logger.error(f"Error running primal dr splitting test: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Testing primal dual dr splitting
        from deblur_denoise.primal_dual_dr.algorithm import test_primal_dual_dr_splitting
        logger.info("Running Primal-Dual Douglas Rachford Splitting algorithm...")
        test_primal_dual_dr_splitting(image_path='/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg',
                                     blur_type='gaussian')
        logger.info("Primal-Dual Douglas Rachford Splitting algorithm completed successfully!")
    except Exception as e:
        logger.error(f"Error running primal dual dr splitting test: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Testing admm 
        from deblur_denoise.admm.algorithm import admm_solver_test
        logger.info("Running ADMM algorithm...")
        admm_solver_test(image_path=IMAGE_PATH,
                         blur_type='gaussian')
        logger.info("ADMM algorithm completed successfully!")
    except Exception as e:
        logger.error(f"Error running admm solver test: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Testing chambolle pock
        from deblur_denoise.chambolle_pock.algorithm import chambolle_pock_test
        logger.info("Running Chambolle-Pock algorithm...")
        chambolle_pock_test(image_path=IMAGE_PATH,
                            blur_type='gaussian')
        logger.info("Chambolle-Pock algorithm completed successfully!")
    except Exception as e:
        logger.error(f"Error running chambolle pock test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting development script")
    reinstall_package()
    run_tests()
    logger.info("Development script completed") 