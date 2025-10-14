This codebase implements the experimental procedures of the paper:
**"Bridging data-rich and data-poor domains on Lithium-Ion Battery via Scanning Electron Microscopic data through Convolutional Neural Network Transfer Learning"**, presented at the AI4Mat-NeurIPS-2025 conference.


## Project Overview
This project compares the performance of EfficientNet-B0 pre-trained models across two distinct image domains to analyze transfer learning and domain adaptation efficacy.


## Requirements and Execution

1.  Matlab: Requires the Deep Learning Toolbox™.
2.  Data: Image data must be organized in class-specific subfolders under the `/data/e1` and `/data/e2` paths.

## Execution Steps

1.  Clone this repository.
2.  Open `main_experiment.m` in Matlab.
3.  Check data paths are correctly configured.
4.  Run the script.

Results, including class-wise performance comparisons and mean confusion matrices, will be displayed in the Command Window and saved to the `confmat_images/` folder.
