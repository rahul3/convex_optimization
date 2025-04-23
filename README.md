# convex_optimization
Blurring and deblurring algorithms.

Implementation of the following algorithms for image deblurring:

1. Primal Douglas-Rachford Splitting (Spingarn’s Method)
2. Primal-Dual Douglas Rachford Splitting
3. ADMM (Dual Douglas-Rachford)
4. Chambolle-Pock Method 


# Build instructions

Enter the `convex_optimization` directory and run:
`pip install -e .`

Then you can import the functions in the project such as

```
from deblur_denoise.chambolle_pock.algorithm import chambolle_pock
```

# Convex Optimization Algorithms

A collection of convex optimization algorithms for image deblurring and denoising, including:

1. Primal Douglas-Rachford Splitting (Spingarn’s Method)
2. Primal-Dual Douglas Rachford Splitting
3. ADMM (Dual Douglas-Rachford)
4. Chambolle-Pock Method 

## Installation

```bash
pip install -e .
```

## Development Setup

### Prerequisites
- Python 3.11.10
- pip
- Virtual environment (recommended)
- Required packages (from requirements.txt):
  - numpy>=1.21.0
  - scipy>=1.7.0
  - matplotlib>=3.4.0
  - torch (for GPU acceleration - TBA)

### Project Structure
```
convex_optimization/
├── deblur_denoise/
│   ├── admm/              # ADMM algorithm implementation
│   │   ├── algorithm.py
│   │   └── __init__.py
│   ├── primal_dr/         # Primal Douglas-Rachford implementation
│   │   ├── algorithm.py
│   │   └── __init__.py
│   ├── primal_dual_dr/    # Primal-Dual Douglas-Rachford implementation
│   │   ├── algorithm.py
│   │   └── __init__.py
│   ├── chambolle_pock/    # Chambolle-Pock implementation
│   │   ├── algorithm.py
│   │   └── __init__.py
│   ├── core/              # Core functionality
│   │   ├── blur.py
│   │   ├── convolution.py
│   │   ├── loss.py
│   │   ├── noise.py
│   │   ├── proximal_operators.py
│   │   └── __init__.py
│   ├── op_math/           # Mathematical operators
│   │   ├── python_code/
│   │   ├── matlab_code/
│   │   └── __init__.py
│   ├── utils/             # Utility functions
│   │   ├── conv_utils.py
│   │   ├── logging_utils.py
│   │   ├── sample_images/
│   │   └── __init__.py
│   ├── blur_deblur.py     # Main deblurring functionality
│   └── __init__.py
├── .vscode/               # IDE configuration
│   ├── launch.json
│   └── tasks.json
├── build_test_script.py   # Development and testing script
├── setup.py               # Package configuration
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
├── Project.md             # Project documentation
└── README.md              # This file
```

### Development Workflow

1. **Install in Development Mode**
   ```bash
   pip install -e .
   ```

2. **Running Tests**
   You can run tests in several ways:

   a. Using the build task (recommended):
   - Press `Cmd+Shift+B` (Mac) or `Ctrl+Shift+B` (Windows/Linux)
   - This will automatically:
     - Uninstall the current package
     - Reinstall it in development mode
     - Run the test suite
     - The environment variable `CO_IMAGE_PATH` can be set to specify the image path for testing

   b. Using the command line:
   ```bash
   python build_test_script.py
   ```

   c. Using the debugger:
   - Open the Run and Debug view (`Cmd+Shift+D` or `Ctrl+Shift+D`)
   - Select "Python: Reinstall and Test"
   - Press F5 to start debugging

### Test Suites
The development script tests all algorithms for a sample image whose path can be read from the environment variable `CO_IMAGE_PATH`.

### Development Tools
- **Build System**: Uses `setup.py` for package management
- **Testing**: Custom test suite with metrics and visualization
- **Debugging**: VS Code/Cursor configuration for easy debugging

### Best Practices
1. Always run the build task after making changes
2. Check test output for any errors
3. Use the debugger for complex issues
4. Keep dependencies up to date in `requirements.txt`
5. Use proper tensor operations (clone/detach) when working with PyTorch tensors
6. Document any changes to algorithm parameters

### Pulling Changes
To update a specific file from master:
```bash
git checkout master -- path/to/file
```

## License
MIT License

## Authors
- Alexander Kakabadze
- Edmand Yu
- Lilian Yuan
- Rahul Padmanabhan
