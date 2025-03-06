# Convex Optimization Algorithms

This package implements various convex optimization algorithms as part of a course project (MATH 463/563).

## Project Structure

The project is organized into four main modules, each implementing a different optimization algorithm:

1. `primal_dr`: Primal Douglas-Rachford Splitting (Lilian Yuan)
2. `primal_dual_dr`: Primal-Dual Douglas-Rachford Splitting (Alexander Kakabadze)
3. `admm`: ADMM (Dual Douglas-Rachford) (Edmand Yu)
4. `chambolle_pock`: Chambolle-Pock Method (Rahul Padmanabhan)

## Installation

```bash
pip install convex-optimization
```

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Building the Package

To build the package:

```bash
python setup.py sdist bdist_wheel
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.