# convex_optimization
Convex Optimization Project (MATH 463/563)
## Contributing to this Project

To contribute to this project, please follow these steps:

1. **Fork the Repository**
   - Click the "Fork" button in the top-right corner of this repository
   - This will create a copy of the repository in your GitHub account

2. **Clone your Fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/convex_optimization.git
   cd convex_optimization
   ```

3. **Create a New Branch**
   ```bash
   git checkout -b your-feature-branch
   ```
   - Name your branch something descriptive related to your changes

4. **Make Your Changes**
   - Work on your changes in your feature branch
   - Commit your changes with clear, descriptive commit messages:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin your-feature-branch
   ```

6. **Create a Pull Request**
   - Go to the [original repository](https://github.com/rahul3/convex_optimization)
   - Click "Pull Requests" and then "New Pull Request"
   - Click "compare across forks"
   - Select your fork and branch as the source
   - Write a clear description of your changes
   - Submit the pull request

### Best Practices
- Keep your fork [synced with the upstream repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
- Create a new branch for each feature or fix
- Write clear commit messages
- Test your changes before submitting a pull request
- Respond to any feedback on your pull request

For more detailed information about working with forks, see GitHub's guide on [working with forks](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks).

# Build instructions

Enter the `convex_optimization` directory and run:
`pip install -e .`

Then you can import the functions in the project such as

```
from deblur_denoise.chambolle_pock.algorithm import chambolle_pock
```

# Convex Optimization Algorithms

A collection of convex optimization algorithms for image deblurring and denoising, including:
- ADMM (Alternating Direction Method of Multipliers)
- Primal Douglas-Rachford Splitting
- Chambolle-Pock Algorithm

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
  - torch (for GPU acceleration)

### Project Structure
```
convex_optimization/
├── deblur_denoise/
│   ├── admm/              # ADMM algorithm implementation
│   ├── primal_dr/         # Primal Douglas-Rachford implementation
│   ├── primal_dual_dr/    # Primal-Dual Douglas-Rachford implementation
│   ├── chambolle_pock/    # Chambolle-Pock implementation
│   ├── core/              # Core functionality
│   │   ├── convolution.py
│   │   ├── noise.py
│   │   └── proximal_operators.py
│   ├── op_math/           # Mathematical operators
│   │   └── python_code/
│   └── utils/             # Utility functions
├── .vscode/               # IDE configuration
├── dev_script.py          # Development and testing script
└── setup.py              # Package configuration
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

   b. Using the command line:
   ```bash
   python dev_script.py
   ```

   c. Using the debugger:
   - Open the Run and Debug view (`Cmd+Shift+D` or `Ctrl+Shift+D`)
   - Select "Python: Reinstall and Test"
   - Press F5 to start debugging

### Test Suites
The development script includes:
1. ADMM Solver Test
   - Tests the ADMM algorithm with different parameters
   - Includes metrics calculation (MSE, PSNR)
   - Visual comparison of results

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
