# GTN Project Overview

## Purpose
Implementation of NTN (Newton Tensor Networks) model with MPS (Matrix Product States) and MPO structures. The project focuses on efficient environment caching for tensor network optimization using Newton updates.

## Tech Stack
- **Language**: Python 3.12
- **Main Libraries**: 
  - quimb: Tensor network library
  - torch: PyTorch for tensor operations
  - numpy: Numerical computations

## Project Structure
- `model/`: Core model implementations
  - `NTN.py`: Newton Tensor Network base class
  - `MPS.py`: Matrix Product State implementation
  - `GTN.py`: Main GTN implementation
  - `builder.py`: Network builder utilities
  - `losses.py`: Loss functions
  - `utils.py`: Utility functions
- `test_*.py`: Various test files for different components
- Documentation files (`.md`): Implementation guides and summaries
