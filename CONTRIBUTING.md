# Contributing to Phi-3 Custom Model

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and inclusive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Phi-3-Custom-Model.git
   cd Phi-3-Custom-Model
   ```
3. **Set up** development environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```
4. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Requirements

- Python 3.10+
- 8 GB RAM minimum
- Git

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest tests/test_cli_entry.py -v

# Run with coverage
python -m pytest --cov=cli --cov-report=html
```

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/rbkhan007/phi3-custom-model/issues)
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

### Suggesting Features

1. Open an issue with `[Feature]` prefix
2. Describe the use case
3. Explain the expected behavior

### Code Contributions

1. **Bug fixes**: Reference the issue number
2. **New features**: Discuss in an issue first
3. **Documentation**: Always welcome!

## Pull Request Process

1. **Update tests** if needed
2. **Run the full test suite**:
   ```bash
   python -m pytest
   ```
3. **Update documentation** if adding features
4. **Create PR** with:
   - Clear title and description
   - Reference related issues
   - Add screenshots if UI changes

### PR Checklist

- [ ] Tests pass (`python -m pytest`)
- [ ] Code follows style guidelines
- [ ] Documentation updated (if applicable)
- [ ] No secrets or API keys committed
- [ ] Large files (.gguf) not committed

## Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Maximum line length: 100 characters

### Naming Conventions

- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

### Example

```python
def calculate_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vector_a: First vector.
        vector_b: Second vector.
        
    Returns:
        Similarity score between -1 and 1.
    """
    # Implementation
    return similarity
```

## Testing

### Test Structure

```
tests/
├── test_cli_entry.py      # CLI tests
├── test_capabilities.py   # RAG, memory, etc.
├── test_tools.py          # Tool calling
├── test_inference.py      # Inference engine
└── test_optimization.py   # CPU throttle, etc.
```

### Writing Tests

```python
def test_my_feature():
    """Test my new feature."""
    # Arrange
    input_data = "test"
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected
```

## Documentation

### README Updates

- Keep the README up-to-date
- Add examples for new features
- Update the project structure if adding modules

### Code Documentation

- Add docstrings to all public functions
- Include type hints
- Explain complex algorithms

## Questions?

Open an issue with `[Question]` prefix or reach out to [@rbkhan007](https://github.com/rbkhan007).

Thank you for contributing! 🎉
