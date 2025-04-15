
# Loan Default Prediction using Genetic Algorithm Optimized Neural Network

## Objective

This project aims to predict loan defaults based on historical loan data (`Wilson_train.csv` and `Wilson_test.csv`). It utilizes a Backpropagation Neural Network (BPNN) whose architecture (specifically, the hidden layer sizes) is optimized using a Genetic Algorithm (GA).

## Dataset

The project uses two datasets located in the `/data` directory:
*   `data/Wilson_train.csv`: Training data containing loan features and a 'Default' target variable.
*   `data/Wilson_test.csv`: Testing data with the same features used for evaluating the trained model.

## Methodology

The core steps involved in this project are:

1.  **Data Loading:** Importing the training and testing datasets using Pandas.
2.  **Data Cleaning & Preprocessing:**
    *   Handling missing values (dropping rows with NaNs in the training set).
    *   Correcting inconsistent data entries (e.g., '0'/'1'/'T' in `Revolving_Credit_Line`).
    *   Parsing and standardizing date formats (`Date_Of_Disbursement`, `Commitment_Date`).
    *   Extracting numerical values from currency strings (e.g., `Guaranteed_Approved _Loan`, `ChargedOff_Amount`).
    *   Converting boolean-like columns (`Code_Franchise`).
3.  **Feature Engineering:**
    *   Calculating the difference in days between the `Commitment_Date` and `Date_Of_Disbursement` (`Days_Difference`).
4.  **Data Transformation:**
    *   Converting date columns into numerical representations (days since the earliest date in the column).
    *   Encoding categorical features using One-Hot Encoding (`sklearn.preprocessing.OneHotEncoder`).
    *   Scaling numerical features using Standardization (`sklearn.preprocessing.StandardScaler`).
    *   Handling potential missing values introduced during processing using imputation (`sklearn.impute.SimpleImputer`).
    *   Applying these transformations using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.
5.  **Model Training & Optimization:**
    *   Defining a function to create and evaluate an `sklearn.neural_network.MLPClassifier` (BPNN).
    *   Using the `deap` library to set up a Genetic Algorithm to search for the optimal number and size of hidden layers for the MLPClassifier.
    *   The fitness function for the GA is the F1-score on the test set.
6.  **Evaluation:**
    *   Training the final MLPClassifier with the best hyperparameters found by the GA.
    *   Evaluating the model performance on the preprocessed test set using a Classification Report and F1-score.

## Results

The Genetic Algorithm explored different hidden layer configurations for the BPNN over 10 generations.

*   **Best Hidden Layer Sizes Found:** `[14, 64, 7]` (Based on the provided notebook output)
*   **Final F1 Score (Weighted):** 0.99 (approximately)
*   **Final F1 Score (Class 1 - Default):** 0.9871 (Based on the provided notebook output)

**Classification Report on Test Data (from notebook):**

precision    recall  f1-score   support

       0       1.00      0.99      0.99     15161
       1       0.98      0.99      0.99      5839

accuracy                           0.99     21000
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

macro avg 0.99 0.99 0.99 21000
weighted avg 0.99 0.99 0.99 21000

## Repository Structure
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

├── data/ # Contains the raw datasets
│ ├── Wilson_train.csv
│ └── Wilson_test.csv
├── notebooks/ # Contains the Jupyter notebook with the analysis
│ └── 4011_task_1.ipynb # Or your chosen notebook name
├── .gitignore # Git ignore file
├── LICENSE # Project License (MIT)
├── README.md # This documentation file
└── requirements.txt # Python dependencies

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    # Replace trenton-lau and loan-default-prediction-ga-bpnn-optimized with your actual details
    git clone https://github.com/trenton-lau/loan-default-prediction-ga-bpnn-optimized.git
    cd loan-default-prediction-ga-bpnn-optimized
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows use: venv\Scripts\activate
    # On macOS/Linux use: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure the data files (`Wilson_train.csv`, `Wilson_test.csv`) are present in the `data/` directory.
2.  Activate your virtual environment if you created one.
3.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
4.  Navigate to the `notebooks/` directory and open the `.ipynb` file.
5.  Run the cells sequentially to execute the data processing, model training, and evaluation steps.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
