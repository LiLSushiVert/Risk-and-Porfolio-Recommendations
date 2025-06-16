# Risk-and-Porfolio-Recommendations

Financial risk analysis and personalized investment portfolio recommendations using Python, machine learning, and hybrid recommendation systems.

 Project Overview

This project aims to:

- Classify individual financial risk using machine learning models (Decision Tree and Random Forest).
- Recommend personalized investment portfolios using a hybrid system combining Rule-Based and Content-Based methods (Score-Level Fusion).
- Analyze and process real-world data from the **Bank Marketing Dataset** with over 4,000 customer records.

 Key Features

 Financial risk classification with high accuracy (>98%) using Decision Tree & Random Forest.
 Personalized investment portfolio recommendation system with 7 risk-based categories.
 Fusion model combining expert rules and data-driven user behavior.
 Performance evaluation using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
 Scalable solution suitable for fintech and banking applications.

 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (DecisionTree, RandomForest)
- Matplotlib, Seaborn
- Streamlit (for interactive visualization, optional)
- Rule-based and Content-based recommendation logic

 Dataset

- **Bank Marketing Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) *(or your data source if different)*
- Includes demographics, account balances, and investment preferences.

 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Risk-and-Portfolio-Recommendations.git
   cd Risk-and-Portfolio-Recommendations
2. Install dependencies:
 pip install -r requirements.txt
3. Run the model:
   python DeAn.py
4.  Launch the interactive app with Streamlit:
    streamlit run app.py
