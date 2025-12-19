# Code Style and Conventions

## General Style
- Python code follows standard Python conventions
- Type hints are not extensively used
- Docstrings are sparse

## Naming Conventions
- Classes: PascalCase (e.g., `NTN`, `MPS_NTN`, `FixedMovingEnvironment`)
- Functions/Methods: snake_case with leading underscore for private methods (e.g., `_batch_forward`, `_batch_environment`)
- Variables: snake_case (e.g., `batch_inds`, `target_tag`)

## Design Patterns
- Class inheritance for specialized tensor network structures (e.g., MPS_NTN extends NTN)
- Tag-based tensor identification in quimb networks
- Environment caching pattern from DMRG algorithms
