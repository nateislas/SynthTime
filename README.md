# **SynthTime: A Platform for Generative Time Series Modeling**

## **Overview**

SynthTime is a Python-based platform designed for generating synthetic time series data using various deep learning models, including TimeGAN. It provides a modular and user-friendly environment for training, evaluating, and deploying these models. The platform emphasizes automated data preprocessing, scenario-based forecasting (in the Conditional TimeGAN variants), and the accurate representation of temporal patterns, with a focus on stocks.

The TimeGAN model is based on the following paper:
- **Yoon, J., Jarrett, D., & van der Schaar, M. (2019).** *Time-series Generative Adversarial Networks*. In *Neural Information Processing Systems (NeurIPS)*. [Link](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks)

---

## **Key Contributions**

### **1. PyTorch Implementation of TimeGAN for Stocks**
* Developed a robust PyTorch implementation of the TimeGAN model to generate synthetic time series, with a particular focus on preserving both temporal patterns and the distributional characteristics of financial data.

### **2. Conditional TimeGAN (ConditionalTimeGAN)**
*   Extended the standard TimeGAN to a **Conditional TimeGAN** (ConditionalTimeGAN). This model generates synthetic sequences based on contextual information, such as:
    *   **Market Regime** (Bull, Bear, Neutral)
    *   **Monthly Return**
*   Utilizes **one-hot encoding** for categorical conditions and continuous scaling for numerical features, enabling scenario-based simulations.

### 3. Conditional TimeGAN for Forecasting (ConditionalTimeGANForecast)

*   Created **ConditionalTimeGANForecast**, an enhanced variant of Conditional TimeGAN specifically for forecasting stock prices.
*   Uses **historical data and conditioning variables** to generate future sequences, offering probabilistic **forecast intervals** (e.g., 5th and 95th percentiles).
*   Focuses on generating accurate and realistic future predictions.

### 4. Unique Features in ConditionalTimeGANForecast

*   Implements a **forecast MSE loss** to refine the generator's ability to predict future sequences accurately.
*   Introduces a **scenario-based forecasting approach**, generating multiple synthetic paths to construct **prediction intervals**.
*   Features an automated pipeline for **data preprocessing**, including **MinMax scaling**, **one-hot encoding**, and efficient sequence generation tailored for forecasting tasks.

---

## **Key Features**

*   **Modular Design:** The platform is designed to be easily extendable to other generative models beyond TimeGAN.
*   **Automated Preprocessing:** Includes efficient handling of scaling, feature encoding, and duplicate removal.
*   **Three-Phase Training:** Implements a robust training pipeline with Autoencoder, Supervisor, and Joint adversarial training phases.
*   **Probabilistic Forecasting:** Generates multiple future scenarios to build **prediction intervals** (available in ConditionalTimeGANForecast).
*   **Comprehensive Evaluation:** Offers tools like PCA and t-SNE for visualization and supports other predictive metrics for data quality validation.
*   **User-Friendly Interface:** Simplifies the process of data loading, model training, and synthetic data generation.
*   **Early Stopping and Save/Load Functionality:** Prevents overfitting and enables easy model deployment.
*   **Synthetic data storage:** The synthetic data is saved in the path `data/synthetic_data`, organized by the name of the dataset and the sequence length. Each sample of the synthetic data is saved in a separated `.csv` file. The default column names are `feature_{i}`, where `i` is the index of the feature.

---

### Prerequisites

*   Python 3.8+
*   PyTorch
*   NumPy
*   Pandas
*   Matplotlib
*   Joblib

You can install the required packages using `pip`:

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/nateislas/SynthTime.git
    cd SynthTime
    ```

2.  (Optional) Set up a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

### Running the Example

1.  **Prepare Your Data:**
    *   Place your time series data into the `data/real_data/` directory.
    *   The example in `timegan_experiment.ipynb` uses a sample time-series dataset. If you want to use your dataset, replace `stock_data.csv`.
2.  **Run the Notebook:**
    *   The primary example is demonstrated in the `timegan_experiment.ipynb` notebook.
    *   Open and run this notebook using Jupyter or JupyterLab.
3.  **Follow the Notebook:**
    *   Run the notebook cells sequentially. The `timegan_experiment.ipynb` notebook demonstrates:
        *   Setting up network parameters.
        *   Loading and preprocessing data.
        *   Training the TimeGAN model.
        *   Generating synthetic time series data.
        *   Evaluating the generated data using visualization techniques.
        *   Inverse Transform of the data.
        *   Saving of the synthetic generated data.
        * 
## Project Structure

The SynthTime project is organized as follows:

*   **`data/`**: Contains the raw and processed data.
    *   **`real_data/`**: Stores the original time series data (e.g., CSV files).
    *   **`synthetic_data/`**: Where synthetic data generated by the models is saved.
*   **`model_checkpoints/`**: Saves trained models and preprocessing scalers.
*   **`src/`**: Contains the core Python source code:
    *   **`data_processing/`**: Modules for data preprocessing.
        *   **`utils.py`**: Includes functions for loading, cleaning, scaling, preparing sequences, setting seeds, and other essential data utilities.
        *   **`signals.py`**: Contains functions for generating technical indicators and signals.
    *   **`models/`**: Contains the deep learning models.
        *   **`TimeGAN_torch/`**: Contains the PyTorch implementations of the generative models.
        *   **`timegan.py`**: Implements the standard TimeGAN model.
            *   **`conditional_timegan.py`**: Implements the Conditional TimeGAN model.
            *   **`conditional_timegan_forecasting.py`**: Implements the forecasting model.
            *   **`metrics/`**: Contains the functions used to evaluate the models.
            *   **`utils.py`**: contains utilities for the models.
*   **`notebooks/`**: Contains Jupyter notebooks that demonstrate how to use the project.
    *   **`timegan_experiment.ipynb`**: An example of how to use the standard TimeGAN model.
    *   **`cond_timegan_experiment.ipynb`**: An example of how to use the Conditional TimeGAN model.
    *  **`cond_timegan_forecasting_experiment.ipynb`**: An example of how to use the forecasting model.
*   **`requirements.txt`**: Lists the Python packages needed to run the project.
*   **`LICENSE`**: Contains the licensing information for the project.
*   **`README.md`**: The file you are currently reading.

**Key files to examine:**
* `src/data_processing/utils.py`: for data loading, scaling, cleaning, and sequence creation.
* `src/models/TimeGAN_torch/timegan.py`: to learn about the core generative model.
* `src/models/TimeGAN_torch/conditional_timegan.py`: to learn about conditional time series generation.
* `src/models/TimeGAN_torch/conditional_timegan_forecasting.py`: to understand the details of probabilistic forecasting.
* `notebooks/timegan_experiment.ipynb`: for an example of training the TimeGAN model.
* `notebooks/cond_timegan_experiment.ipynb`: for an example of training the Conditional TimeGAN model.
* `notebooks/cond_timegan_forecasting_experiment.ipynb`: for an example of training the forecasting model.

## Future Enhancements

*   **More Evaluation Metrics:** Implement a wider range of quantitative evaluation metrics 
*   **Hyperparameter Tuning:** Add functionality for hyperparameter optimization.
*   **Support for More Models:** Expand the platform to include other generative models.