# Change Log

-------------------------
## [Unrelease]
### Added
- Add classes for backward.

### Changed

### Fixed

### Removed

-------------------------
## [1.0.0] - 2022/06/28
### Added
- Implement method `myDataset::load()` to load mnist dataset.
- Add method into the class `myDataset` to get the fields.
- Add the class `myDatasetIterator` to travel a whole dataset.
- Add the class `myNetwork` to predict.
- Update the class `myNetwork` to calculate loss and gradient.
- Update code for training.
- Update the class `myNetwork` to save and load network weights.
- Update test code to save the final weights.

### Changed
- Update the class `myDatasetIterator` for batch training.

### Fixed
- Fixed the bug that the dataset iterator could not move next items.
