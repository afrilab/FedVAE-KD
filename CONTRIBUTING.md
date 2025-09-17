# Contributing to FedVAE-KD

Thank you for your interest in contributing to FedVAE-KD! We welcome contributions from the community to help improve this privacy-preserving federated learning framework for wireless network intrusion detection.

## Getting Started

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Write tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## Code Style

We follow the PEP 8 style guide for Python code. Please ensure your code adheres to these standards:

- Use 4 spaces for indentation (no tabs)
- Limit lines to 79 characters
- Use descriptive variable and function names
- Write docstrings for all public classes and functions
- Import statements should be at the top of the file, grouped and ordered

## Testing

All contributions must include appropriate tests. We use pytest for testing:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_vae.py
```

## Documentation

Please update documentation when making changes to the code:

- Update docstrings in the code
- Update README.md if necessary
- Add comments to explain complex logic

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent
4. Your pull request will be reviewed by maintainers, who may request changes
5. Once approved, your pull request will be merged

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:

- A clear and descriptive title
- A detailed description of the problem or feature
- Steps to reproduce the issue (if applicable)
- Expected behavior vs. actual behavior
- Any relevant code snippets or error messages

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## Questions?

If you have any questions about contributing, feel free to open an issue or contact the maintainers.