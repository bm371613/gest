# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- open pinch annotation implemented

### Changed
- new model: much smaller, much faster, slightly less accurate

### Fixed
- fixed a bug in capturing app: sometimes preview did not display annotations

## [0.2.0] - 2020-10-31
### Changed
- input, preprocessing, inference and postprocessing in separate threads for better responsiveness

## [0.1.0] - 2020-10-21
### Added
- pinching gesture recognition, in one hand orientation
- heatmap output, separate for left and right hand, indicating pinched point position
- demo for testing recognition models
- example script for simulating mouse clicks and scrolling
- scripts for producing and reviewing training data


[Unreleased]: https://github.com/bm371613/gest/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bm371613/gest/releases/tag/v0.2.0
[0.1.0]: https://github.com/bm371613/gest/releases/tag/v0.1.0
