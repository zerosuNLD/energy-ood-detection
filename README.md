# Energy-based OOD Detection

Implementation and analysis of Energy-based Out-of-distribution (OOD) Detection proposed by Liu et al. (2020).

This project evaluates the effectiveness of **Energy Score** for OOD detection and compares it with the **Maximum Softmax Probability (MSP)** baseline on multiple image datasets.

## Features

* Energy-based OOD detection
* MSP baseline comparison
* Evaluation on multiple OOD datasets:

  * SVHN
  * LSUN
  * Tiny-ImageNet
* Temperature scaling analysis
* Automatic generation of ROC, PR, AUROC, AUPR, FPR95, and F1 evaluation results
* Figure generation for reports

## Project Structure

```text
data/                       # Datasets
results/                    # Generated figures and outputs
├── charts/
├── charts_combined/
└── individual/

run_ood_analysis.py         # Main evaluation script
eval_ood_all.py             # Multi-dataset evaluation
evaluate_temperature.py     # Temperature analysis
deep_analysis.py            # Detailed analysis

export_charts.py
export_combined_figures.py
export_individual_figures.py

wrn28_10_cifar10.pth        # Pretrained WRN-28-10
main.tex                    # Report source
TrustWorthyAI.ipynb         # Notebook version
```

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* numpy
* matplotlib
* scikit-learn
* scipy
* tqdm

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy tqdm
```

## Usage

Run the main evaluation:

```bash
python run_ood_analysis.py
```

Evaluate multiple OOD datasets:

```bash
python eval_ood_all.py
```

Evaluate different temperature values:

```bash
python evaluate_temperature.py
```

Generate figures:

```bash
python export_charts.py
python export_combined_figures.py
python export_individual_figures.py
```

Compile the report:

```bash
pdflatex main.tex
```

## Pretrained Model

Download the pretrained WRN-28-10 model and place it in the project root directory:

```bash
wget https://github.com/zerosuNLD/energy-ood-detection/releases/download/v1.0/wrn28_10_cifar10.pth
```

or

```bash
curl -L -o wrn28_10_cifar10.pth \
https://github.com/zerosuNLD/energy-ood-detection/releases/download/v1.0/wrn28_10_cifar10.pth
```

## Reference

W. Liu, X. Wang, J. Owens, and Y. Li.

**Energy-based Out-of-distribution Detection**.

NeurIPS 2020.
