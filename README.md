# Pull-Up Training Analysis Experiment

A comprehensive data analysis project that tracks and analyzes pull-up training progress, examining the relationship between dead-hang training and pull-up performance, as well as body composition changes over time.

## Project Overview

This project uses statistical analysis and interactive visualizations to answer two key research questions:

1. **Does daily dead-hang training improve pull-up performance?**
   - Analyzes correlation between dead-hang duration and pull-up capacity
   - Tracks performance trends over time
   - Provides statistical validation (p-values, R² values)

2. **Are body composition changes associated with pull-up performance?**
   - Examines muscle circumference measurements (forearms, biceps, lats)
   - Tracks weight changes
   - Correlates physical changes with performance improvements


## Technology Stack

- **Python 3.14+**
- **Marimo**: Interactive notebook environment
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **NumPy**: Numerical computations
- **SciPy**: Statistical analysis
- **Scikit-learn**: Machine learning utilities
- **uv**: Fast Python package manager

## Installation

### Prerequisites

- Python 3.14 or higher
- pip (Python package manager)

### Option 1: Using uv (Recommended - Fast!)

1. **Install uv** (if not already installed):

   **Windows:**
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd CCDATSCL_PROJECT_COM222
   ```

3. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

   This will automatically create a virtual environment and install all dependencies from `uv.lock`.

### Option 2: Using pip

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd CCDATSCL_PROJECT_COM222
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Notebook

**With uv:**
```bash
uv run marimo edit pullup_file.py
```

**With pip (after activating venv):**
```bash
marimo edit pullup_file.py
```

This will open the marimo notebook in your default web browser (typically at `http://localhost:2718`).

### Running as a Static App

**With uv:**
```bash
uv run marimo run pullup_file.py
```

**With pip:**
```bash
marimo run pullup_file.py
```

### Viewing Saved Outputs

The cleaned data is automatically saved to `data/pullup_logs_cleaned.csv` after running the analysis.

## Project Structure

```
CCDATSCL_PROJECT_COM222/
├── data/
│   ├── pullup_logs.csv              # Raw training data
│   ├── pullup_logs_cleaned.csv      # Processed data
│   └── weekly_data/                 # Weekly breakdown data
│       ├── week_1.csv
│       ├── week_2.csv
│       ├── week_3.csv
│       ├── week_4.csv
│       ├── week_5.csv
│       └── week_6.csv
├── __marimo__/
│   └── session/
│       └── pullup_file.py.json      # Marimo session config
├── pullup_file.py                   # Main analysis dashboard
├── main.py                          # Alternative entry point
├── pyproject.toml                   # Project metadata
├── requirements.txt                 # Python dependencies
├── uv.lock                          # Locked dependencies (uv)
└── README.md                        # This file
```

## 📊 Data Format

The input data (`pullup_logs.csv`) should contain the following columns:

- **Date**: Training date (MM/DD/YYYY)
- **Average Dead Hang (secs)**: Average dead-hang duration
- **Maximum Pull-Ups**: Maximum consecutive pull-ups achieved
- **Left/Right Forearm Circumference (cm)**: Forearm measurements
- **Left/Right Biceps Circumference (cm)**: Biceps measurements
- **Lats Spread Width (cm)**: Back width measurement
- **Weight (kg)**: Body weight
- **Perceived Difficulty (1-10)**: Subjective difficulty rating

## Analysis Output

The analysis provides:

1. **Dataset Overview**: Summary statistics and date ranges
2. **Research Question 1 Analysis**:
   - Correlation coefficients
   - Trend analysis with R² values
   - Performance improvement rates
   - Interactive visualizations
3. **Research Question 2 Analysis**:
   - Body composition correlations
   - Muscle growth tracking
   - Statistical significance tests
4. **Additional Insights**:
   - Correlation heatmaps
   - Distribution plots
   - Perceived difficulty trends

## Statistical Methods

- **Pearson Correlation**: Measures linear relationship strength
- **Linear Regression**: Models trends over time
- **P-value Testing**: Determines statistical significance (α = 0.05)
- **R² (Coefficient of Determination)**: Explains variance in the data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is created for academic purposes as part of CCDATSCL coursework.

## Author

**Rodney Lei Estrada**

## Acknowledgments

- **Marimo**: For providing an excellent interactive notebook environment
- **Astral (uv)**: For the blazing-fast package manager
- **SciPy & NumPy communities**: For robust scientific computing tools

## Support

For issues or questions, please open an issue in the repository.

---

**Note**: Make sure your training data is placed in the `data/` directory before running the analysis.