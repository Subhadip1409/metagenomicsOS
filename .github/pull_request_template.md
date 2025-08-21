## 📋 Pull Request Summary

### Type of Change

<!-- Mark the appropriate option with an [x] -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation (changes to documentation only)
- [ ] 🧹 Code cleanup (refactoring, formatting, etc.)
- [ ] 🚀 Performance improvement
- [ ] 🔧 Configuration change
- [ ] ✅ Test addition or update

### Description

<!-- Provide a clear and concise description of your changes -->

### Related Issues

<!-- Link to related issues using keywords: Fixes #123, Closes #456, Resolves #789 -->

- Fixes #
- Related to #

## 🧪 Testing

### Testing Performed

<!-- Describe the tests you ran and how to reproduce them -->

- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] Integration tests pass (`pytest tests/integration/`)
- [ ] Manual testing performed
- [ ] End-to-end workflow testing

### Test Details
Commands used for testing

pytest tests/unit/test_new_feature.py -v

metagenomicsOS run qc --input test_data/ --dry-run


### Test Results
<!-- Paste relevant test output or describe results -->



## 📝 Changes Made

### Files Changed
<!-- List the main files modified and briefly explain why -->
- `src/metagenomicsOS/core/new_module.py` - Added new functionality for X
- `tests/unit/test_new_module.py` - Unit tests for new functionality
- `docs/user-guide/advanced-features.md` - Documentation for new feature

### Code Changes
<!-- Highlight important code changes or design decisions -->


### Configuration Changes
<!-- List any changes to config files, dependencies, or environment -->
- [ ] Added new dependencies (list in pyproject.toml)
- [ ] Modified configuration schema
- [ ] Updated environment requirements
- [ ] Changed default settings

## 📖 Documentation

### Documentation Updated
- [ ] Code comments and docstrings
- [ ] README.md
- [ ] User guide
- [ ] API documentation
- [ ] Developer documentation
- [ ] Changelog updated

### Usage Examples
<!-- If applicable, provide usage examples -->

### Example of new functionality

from metagenomicsOS import new_feature
result = new_feature.analyze(data)


## ✅ Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added type hints where applicable

### Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

### Documentation
- [ ] I have made corresponding changes to the documentation
- [ ] My changes are covered in the user guide (if user-facing)
- [ ] API documentation is updated (if applicable)

### Backwards Compatibility
- [ ] My changes maintain backwards compatibility
- [ ] If breaking changes exist, I have documented the migration path
- [ ] I have updated version numbers appropriately

## 🔗 Additional Context

### Screenshots (if applicable)
<!-- Add screenshots for UI changes -->

### Performance Impact
<!-- Describe any performance implications -->
- Memory usage:
- CPU impact:
- Storage requirements:

### Deployment Notes
<!-- Any special considerations for deployment -->


### Breaking Changes
<!-- Detail any breaking changes and migration steps -->


---

**Reviewer Notes:**
<!-- For maintainers: add any specific review focus areas -->

/cc @maintainer-username
