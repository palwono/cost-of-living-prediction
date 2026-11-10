# FIK FAIR 2026: Cost of Living Prediction 🏙️💰

This repository contains the official inference pipeline and scientific report for the **Data Science Competition FIK FAIR 2026**.

**Team Name:** Prikitiw  
**Status:** Participant  
**Track:** Data Science

### 📌 Project Objective
The objective is to predict the Anonymous Competition Score (ACS), an indicator of regional cost of living, utilizing a highly constrained and fully anonymized dataset. With only 382 training observations and 24 features available, accurate forecasting in this micro-dataset environment demands heavy regularization and strict prevention of variance explosion.

### 🧠 Methodology & Architecture
*   **Algorithm Paradigm:** Shifted from traditional tree-based models to **TabPFN v2**, a state-of-the-art tabular foundation model utilizing in-context learning, which is proven to be highly robust on small datasets.
*   **Modeling Strategy:** Architected a **Dual-Representation Multi-Seed Ensemble (`T3_TA12_50`)**:
    *   **Multi-Seed Stochastic Averaging ($T_3$):** Averaged three raw-feature TabPFN models across different seeds (42, 17, 2026) to reduce internal in-context transformer variance.
    *   **Orthogonal Manifold Augmentation ($TA_{12}$):** Extracted 12 Principal Components (PCA) explaining 99.2% of the variance, concatenated with the original 24 features to provide global spatial regularization.
    *   **Final Fusion:** Weighted combination allocating 50% to the 3-seed raw models ($1/6$ each) and 50% to the PCA-augmented model.
*   **Data Integrity:** Maintained absolute zero target leakage. PCA was fitted strictly on training folds. Purposely avoided artificial post-processing or threshold clipping to ensure stability against private leaderboard shakeups.

### 🚀 Results & Evaluation
*   **Kaggle Public Leaderboard RMSE:** **8.03913**
*   **Kaggle Private Leaderboard RMSE:** **8.15130**
*   **Final Final Score:** **34.62**

### 📁 Repository Structure
*   `data/`: Contains the raw datasets (Train, Test, Submission Template, Data Dictionary).
*   `docs/`: Contains the formal scientific report (`.pdf`).
*   `notebooks/`: The main Jupyter Notebook detailing the entire inference pipeline, factory components, and architecture (`FINAL.ipynb`).
*   `submissions/`: The finalized `.csv` submission file ready for evaluation.

### ⚙️ How to Run
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook in the `notebooks/` directory sequentially.