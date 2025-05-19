# 🧬 Genetic-Algorithm-Based-Feature-Selection-for-Optimizing-Credit-Risk-Algorithms

This project implements a Genetic Algorithm (GA) for feature selection applied to credit risk prediction (Home Credit Default Risk dataset). It integrates:

- **Mutual Information & Total Correlation (TC)** for redundancy-aware fitness evaluation  
- **Logistic Regression** for model validation  
- **Entropy-based redundancy reduction**  
- **Synthetic feature generation**  
- **Tournament selection, fixed-k binary encoding, and constrained mutation/crossover**

---

## 📁 Dataset

This implementation uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.  
Download the `application_train.csv` and update the path in `main()`.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/genetic-feature-selector.git
cd genetic-feature-selector
```

2. (Optional) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

Make sure the dataset is available locally and the path is correctly set inside `main()` in `genetic_feature_selector.py`.

Then run:

```bash
python genetic_feature_selector.py
```

---

## 📊 Output

- Best-performing feature combination is printed at the end
- Visualizations of:
  - ROC AUC over generations
  - Accuracy over generations
  - Fitness (Mutual Information + Total Correlation) over generations

---

## 📦 Requirements

Save this in `requirements.txt`:

```
numpy
pandas
scikit-learn
scipy
matplotlib
```

---



