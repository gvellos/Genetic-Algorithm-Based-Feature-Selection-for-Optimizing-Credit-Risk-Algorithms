import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter
from dit import Distribution
from dit.multivariate import total_correlation





# Load data
app_train = pd.read_csv('/Users/george_vellos/Desktop/INFORMATICS_UNIPI/ΠΤΥΧΙΑΚΗ/home-credit-default-risk/application_train.csv')

# Sample the data for efficiency
app_train = app_train.sample(frac=0.5, random_state=42).reset_index(drop=True)
print(len(app_train), "rows and", len(app_train.columns), "columns")    

# Clean data (Handle NaNs, replace invalid values)
app_train.replace({365243: np.nan, 'XNA': np.nan}, inplace=True)
app_train.replace([np.inf, -np.inf], np.nan, inplace=True)

# Add new features
app_train['CREDIT_TO_INCOME_RATIO'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
app_train['AMT_PAY_YEAR'] = app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY']
app_train['AGE_EMPLOYED'] = app_train['DAYS_EMPLOYED'] - app_train['DAYS_BIRTH']
app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']

# One-hot encode categorical variables
app_train = pd.get_dummies(app_train, columns=app_train.select_dtypes(include=['object']).columns)

# Replace infinities with NaN, then fill missing values
app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
app_train.fillna(app_train.median(), inplace=True)

# Standardize the dataset (excluding target)
target_column = 'TARGET'
feature_columns = [col for col in app_train.columns if col != target_column]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(app_train[feature_columns])
scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns)

# Reattach the target column for modeling
scaled_features_df[target_column] = app_train[target_column]

# ---- Genetic Algorithm Functions ---- #

# Compute entropy for a single feature
def compute_entropy(data):
    """Computes entropy of a single variable."""
    value_counts = data.value_counts(normalize=True)  # Get probability distribution
    return entropy(value_counts, base=2)  # Compute Shannon entropy

# Compute joint entropy for multiple features
def compute_joint_entropy(df):
    """Computes joint entropy for multiple variables in a DataFrame."""
    joint_counts = df.value_counts(normalize=True)  # Joint probability distribution
    return entropy(joint_counts, base=2)  # Compute Shannon entropy

# Compute total correlation (TC) of features
def compute_total_correlation(features_df):
    """Computes total correlation: TC(X, Y, Z, ...) = sum H(Xi) - H(X1, X2, ..., Xn)."""
    individual_entropies = sum(compute_entropy(features_df[col]) for col in features_df.columns)
    joint_entropy = compute_joint_entropy(features_df)
    return individual_entropies - joint_entropy


def fitness_function(features, train_data, target_column):
    if not features:
        return -np.inf  # Invalid chromosome (no features selected)

    X = train_data[features]
    y = train_data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=1)

    # Generate first new feature (_combo_dev)
    new_feature_1 = "_x_".join(features) + "_combo_dev"
    X_train[new_feature_1] = (
        X_train[features].prod(axis=1) +
        X_train[features].std(axis=1) -
        X_train[features].mean(axis=1)
    )
    X_val[new_feature_1] = (
        X_val[features].prod(axis=1) +
        X_val[features].std(axis=1) -
        X_val[features].mean(axis=1)
    )


    # Generate second new feature (_pairwise_difference)
    new_feature_2 = "_x_".join(features) + "_pairwise_difference"
    X_train[new_feature_2] = (
        X_train[features].prod(axis=1) +
        X_train[features].sum(axis=1).sub(X_train[features].mean(axis=1)) +
        (X_train[features].mean(axis=1) * len(features))
    )
    X_val[new_feature_2] = (
        X_val[features].prod(axis=1) +
        X_val[features].sum(axis=1).sub(X_val[features].mean(axis=1)) +
        (X_val[features].mean(axis=1) * len(features))
    )



    new_feature_3 = "_x_".join(features) + "_normalized_deviation"
    X_train[new_feature_3] = (
        X_train[features].sub(X_train[features].mean(axis=1), axis=0).abs().sum(axis=1) / len(features)
    )
    X_val[new_feature_3] = (
        X_val[features].sub(X_val[features].mean(axis=1), axis=0).abs().sum(axis=1) / len(features)
    )





    # X_combined = X_train[[new_feature_1, new_feature_2, new_feature_3]]
    # X_combined = X_combined.copy()
    # mic = mutual_info_classif(X_combined, y_train)

    # # Store mutual information scores
    # feature_MI_score = pd.Series(mic, index=X_combined.columns)

    # score = feature_MI_score[X_combined.columns].sum()



    # Select only valid numerical features for total correlation
    selected_features_df = X_train[features + [new_feature_1, new_feature_2, new_feature_3]]


    # print(selected_features_df.columns)
    # Compute total correlation score
    score = compute_total_correlation(selected_features_df)

    return score



def initialize_population(num_features, population_size, k, important_features=set()):
    """ Ensure each chromosome has exactly k selected features """
    population = []
    for _ in range(population_size):
        chromosome = [0] * num_features
        selected_indices = set(random.sample(range(num_features), k))

        # Prioritize important features if provided
        selected_indices.update(random.sample(list(important_features), min(len(important_features), k)))

        # Ensure exactly k ones
        selected_indices = list(selected_indices)[:k]
        for idx in selected_indices:
            chromosome[idx] = 1
        
        population.append(chromosome)

    return population


def binary_crossover(parent1, parent2):
    """ Two-point crossover while ensuring exactly k ones """
    length = len(parent1)
    k = sum(parent1)  # Fixed number of selected features

    # Perform two-point crossover
    point1, point2 = sorted(random.sample(range(length), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    # Ensure exactly k ones in both children
    def fix_chromosome(chromosome):
        ones = [i for i, bit in enumerate(chromosome) if bit == 1]
        zeros = [i for i, bit in enumerate(chromosome) if bit == 0]
        
        # Too many ones → randomly remove extras
        while len(ones) > k:
            remove_idx = random.choice(ones)
            chromosome[remove_idx] = 0
            ones.remove(remove_idx)
        
        # Too few ones → randomly add missing ones
        while len(ones) < k:
            add_idx = random.choice(zeros)
            chromosome[add_idx] = 1
            zeros.remove(add_idx)
            ones.append(add_idx)

        return chromosome

    return fix_chromosome(child1), fix_chromosome(child2)


def binary_mutation(chromosome, mutation_rate=0.3):
    """ 
    Swap a 1 → 0 and a 0 → 1 based on a mutation rate, 
    while maintaining exactly k ones in the chromosome.
    """
    if random.random() < mutation_rate:  # Apply mutation with probability `mutation_rate`
        ones = [i for i, bit in enumerate(chromosome) if bit == 1]
        zeros = [i for i, bit in enumerate(chromosome) if bit == 0]

        if ones and zeros:
            remove_idx = random.choice(ones)
            add_idx = random.choice(zeros)

            chromosome[remove_idx] = 0
            chromosome[add_idx] = 1

    return chromosome








def tournament_selection(population, fitness_scores, num_parents, tournament_size=3):
    """ Selects best individuals via tournament selection """
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(fitness_scores, population)), tournament_size)
        winner = max(tournament, key=lambda x: x[0])
        parents.append(winner[1])
    return parents

scoresGA = [] 
scoresMI = []
scoresACC = []
def genetic_algorithm(train_data, target_column, feature_names, generations, population_size, tournament_size):
    num_features = len(feature_names)
    k = 67  # Number of selected features 
    population = initialize_population(num_features, population_size, k)

    best_logistic_features = set()  # Store best features found by logistic regression

    for generation in range(generations):
        # Convert binary chromosomes into selected feature lists
        selected_features_list = [
            [feature_names[i] for i in range(num_features) if chrom[i] == 1] 
            for chrom in population
        ]

        # Evaluate fitness
        fitness_scores = [fitness_function(features, train_data, target_column) for features in selected_features_list]

        # Tournament selection for parents
        parents = tournament_selection(population, fitness_scores, num_parents=population_size // 2, tournament_size=tournament_size)

        # Generate next generation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = binary_crossover(parent1, parent2)
            next_generation.append(binary_mutation(child1))
            if len(next_generation) < population_size:
                next_generation.append(binary_mutation(child2))

        # Update population
        population = next_generation

        # Find the best chromosome in this generation
        best_idx = np.argmax(fitness_scores)
        best_score = fitness_scores[best_idx]
        best_combination = selected_features_list[best_idx]

        print(f"Generation {generation + 1}, Best Fitness Score: {best_score}")
        # print(f"Best Combination: {best_combination}")
        print(f"Number of Features: {len(best_combination)}")

        new_feature_1 = "_x_".join(best_combination) + "_combo_dev"
        new_feature_2 = "_x_".join(best_combination) + "_pairwise_difference"
        new_feature_3 = "_x_".join(best_combination) + "_normalized_deviation"

        new_features = pd.DataFrame({
        new_feature_1: (
            train_data[best_combination].prod(axis=1) +
            train_data[best_combination].std(axis=1) -
            train_data[best_combination].mean(axis=1)
        ),
        new_feature_2: (
            train_data[best_combination].prod(axis=1) +
            train_data[best_combination].sum(axis=1).sub(train_data[best_combination].mean(axis=1)) +
            (train_data[best_combination].mean(axis=1) * len(best_combination))
        ),
        new_feature_3: (
            train_data[best_combination].sub(train_data[best_combination].mean(axis=1), axis=0).abs().sum(axis=1) / len(best_combination)
        )
        })

        train_data = pd.concat([train_data, new_features], axis=1)






        

        # Train Logistic Regression on Best Combination
        X_selected = train_data[best_combination]  
        y = train_data[target_column]

        # Ensure the indices are aligned properly
        common_indices = X_selected.index.intersection(y.index)
        X_selected = X_selected.loc[common_indices]
        y = y.loc[common_indices]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.30, random_state=1)

        # Standardize the selected features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        scoresGA.append(roc_auc)
        scoresACC.append(accuracy)
        scoresMI.append(best_score)

        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        print(f"Logistic Regression ROC AUC Score: {roc_auc:.4f}")
        print("\n")

        # Extract feature importance (coefficients) from logistic regression
        feature_importance = abs(model.coef_[0])  # Get absolute values of coefficients
        sorted_indices = np.argsort(feature_importance)[::-1]  # Sort features by importance

        # Select top features with nonzero coefficients
        significant_features = [best_combination[i] for i in sorted_indices if feature_importance[i] > 0]   
        
        # print(f"Significant Features Found by Logistic Regression: {significant_features}")
        # print(f"Number of Significant Features: {len(significant_features)}")

        # Store best features found by logistic regression
        best_logistic_features.update(significant_features)




    # Return the best feature combination
    best_chromosome = population[np.argmax(fitness_scores)]
    return [feature_names[i] for i in range(num_features) if best_chromosome[i] == 1]


# ---- Run Genetic Algorithm ---- #
best_features = genetic_algorithm(
    train_data=scaled_features_df,
    target_column='TARGET',
    feature_names=feature_columns,
    generations=100,
    population_size=40,
    tournament_size=3
)

print("Best Feature Combination:", best_features)

# Define the number of generations (assuming all lists have the same length)
generations = range(1, len(scoresGA) + 1)

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows, 1 column

# Plot ROC AUC Score
axes[0].plot(generations, scoresGA, marker='o', linestyle='-', color='tab:blue')
axes[0].set_title("ROC AUC Score Over Generations")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("ROC AUC Score")
axes[0].grid(True)

# Plot Accuracy
axes[1].plot(generations, scoresACC, marker='s', linestyle='-', color='tab:orange')
axes[1].set_title("Accuracy Over Generations")
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True)

# Plot Mutual Information (uncomment if you have scoresMI)
axes[2].plot(generations, scoresMI, marker='^', linestyle='-', color='tab:green')
axes[2].set_title("Mutual Information Over Generations")
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Mutual Information Score")
axes[2].grid(True)

# Adjust layout and show the plots
fig.suptitle("Genetic Algorithm Performance Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the main title
plt.show()
