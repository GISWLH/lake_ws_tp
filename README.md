# Lake Water Storage Attribution Model

## Project Overview

This project implements a lake water storage attribution model using TabPFN (Tabular Prior-data Fitted Networks), a pre-trained tabular tree model similar to Random Forest. The model analyzes and predicts lake water volume changes based on various environmental and anthropogenic factors.

## Features

- **Machine Learning Model**: TabPFN regressor for water volume change prediction
- **Comprehensive Feature Set**: 25 environmental and anthropogenic variables
- **Data Processing**: Automated data preprocessing and cleaning
- **Visualization**: Matplotlib-based plotting with customizable styling
- **Partial Dependence Analysis**: Tools for model interpretation and feature importance

## Project Structure

```
lake_WS_TP/
├── data/                           # Data directory
│   ├── cleaned_data.csv           # Processed dataset
│   ├── data.csv                   # Raw dataset
│   ├── prediction_results_cleaned.csv
│   └── total.xlsx                 # Original data file
├── model/                         # Model directory
│   └── tabpfn-v2-regressor.ckpt  # Pre-trained model checkpoint
├── code/                          # Source code directory
├── data_pre_process.py            # Data preprocessing script
├── partial_dependence_*.py        # Partial dependence analysis scripts
├── test.py                        # Testing script
├── test_new.ipynb               # Jupyter notebook for testing
├── tab_environment_*.yml          # Conda environment files
└── README.md                      # This file
```

## Variables

### Target Variable
- **Water_Volumn_Change**: Lake water storage change (dependent variable)

### Feature Variables (Independent Variables)
| Environmental | Meteorological | Land Use | Anthropogenic |
|---------------|----------------|----------|---------------|
| ET (Evapotranspiration) | PRECIPITATION | Cropland_Area | Population_Density |
| GPP (Gross Primary Productivity) | Temperature | Forest_Area | Human Influence Index |
| LST_DAY (Land Surface Temperature) | Wind_Speed | Steppe_Area | CO2 |
| NPP (Net Primary Productivity) | Vap (Vapor Pressure) | Non-Vegetated/Artificial Land_Area | CH4 |
| Soil_Moisture | Snow_Cover | Wetland_Area | N2O |
| SRAD (Solar Radiation) | Nighttime | Snow/Ice_Area | SF6 |
| NDVI (Normalized Difference Vegetation Index) | | | |

## Installation

### Prerequisites
- Anaconda or Miniconda
- Python 3.10+

### Environment Setup

1. Clone the repository:
```bash
git clone git@github.com:GISWLH/lake_ws_tp.git
cd lake_ws_tp
```

2. Create and activate the conda environment:

**For Windows:**
```bash
conda env create -f tab_environment_windows.yml
conda activate tab
```

**For Linux/macOS:**
```bash
conda env create -f tab_environment_no_builds.yml
conda activate tab
```

### Key Dependencies
- **TabPFN**: 2.0.9 (Main ML model)
- **TabPFN Extensions**: 0.0.4
- **SHAP**: 0.47.2 (Model interpretation)
- **Scikit-learn**: Machine learning utilities
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **XGBoost & CatBoost**: Additional ML models
- **PyTorch**: Deep learning framework

## Usage

### Data Preprocessing
```bash
conda activate tab
python data_pre_process.py
```

### Model Training and Prediction
```bash
python test.py
```

### Jupyter Notebook Analysis
```bash
jupyter notebook test_new.ipynb
```

### Partial Dependence Analysis
```bash
python partial_dependence_complete_fixed.py
```

## Important Notes

- **Environment**: Always activate the `tab` conda environment before running any scripts
- **Execution Time**: Model training and analysis may take 5+ minutes due to computational complexity
- **Visualization**: All plots use Arial font and English labels
- **Code Style**: Direct, Jupyter notebook-friendly code structure with minimal complex function definitions

## Model Performance

The model analyzes the relationship between environmental factors and lake water storage changes, providing insights into:
- Climate change impacts on water resources
- Human activity effects on lake systems
- Seasonal and temporal water storage patterns
- Feature importance for water volume prediction

## Contributing

This project follows a Jupyter notebook-oriented development approach with emphasis on:
- Direct, readable code
- Minimal complex function definitions
- Clear variable naming
- Comprehensive documentation

## License

[Add your license information here]

## Contact

[Add your contact information here]

## Acknowledgments

- TabPFN development team for the pre-trained model
- Environmental data providers
- Research community contributions
