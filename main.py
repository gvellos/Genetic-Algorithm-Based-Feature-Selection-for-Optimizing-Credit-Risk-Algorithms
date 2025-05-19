import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
import matplotlib.pyplot as plt

class GeneticFeatureSelector:
    def __init__(self, data_path, target_column='TARGET', k=73, generations=100, population_size=40, tournament_size=3):
        self.data_path = data_path
        self.target_column = target_column
        self.k = k
        self.generations = generations
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.scoresGA = []
        self.scoresMI = []
        self.scoresACC = []

    def load_and_prepare_data(self):
        app_train = pd.read_csv(self.data_path)
        app_train.replace({365243: np.nan, 'XNA': np.nan}, inplace=True)
        app_train.replace([np.inf, -np.inf], np.nan, inplace=True)

        app_train['CREDIT_TO_INCOME_RATIO'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
        app_train['AMT_PAY_YEAR'] = app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY']
        app_train['AGE_EMPLOYED'] = app_train['DAYS_EMPLOYED'] - app_train['DAYS_BIRTH']
        app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']

        app_train = pd.get_dummies(app_train, columns=app_train.select_dtypes(include=['object']).columns)
        app_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        app_train.fillna(app_train.median(), inplace=True)

        self.feature_columns = [col for col in app_train.columns if col != self.target_column]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(app_train[self.feature_columns])
        self.scaled_features_df = pd.DataFrame(scaled_features, columns=self.feature_columns)
        self.scaled_features_df[self.target_column] = app_train[self.target_column]

    def compute_entropy(self, data):
        value_counts = data.value_counts(normalize=True)
        return entropy(value_counts, base=2)

    def compute_joint_entropy(self, df):
        joint_counts = df.value_counts(normalize=True)
        return entropy(joint_counts, base=2)

    def compute_total_correlation(self, features_df):
        individual_entropies = sum(self.compute_entropy(features_df[col]) for col in features_df.columns)
        joint_entropy = self.compute_joint_entropy(features_df)
        return individual_entropies - joint_entropy

    def fitness_function(self, features):
        if not features:
            return -np.inf

        X = self.scaled_features_df[features]
        y = self.scaled_features_df[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=1)

        new_feature_1 = "_x_".join(features) + "_combo_dev"
        X_train[new_feature_1] = X_train[features].prod(axis=1) + X_train[features].std(axis=1) - X_train[features].mean(axis=1)
        X_val[new_feature_1] = X_val[features].prod(axis=1) + X_val[features].std(axis=1) - X_val[features].mean(axis=1)

        new_feature_2 = "_x_".join(features) + "_pairwise_difference"
        X_train[new_feature_2] = X_train[features].prod(axis=1) + X_train[features].sum(axis=1) - X_train[features].mean(axis=1) * len(features)
        X_val[new_feature_2] = X_val[features].prod(axis=1) + X_val[features].sum(axis=1) - X_val[features].mean(axis=1) * len(features)

        new_feature_3 = "_x_".join(features) + "_normalized_deviation"
        X_train[new_feature_3] = X_train[features].sub(X_train[features].mean(axis=1), axis=0).abs().sum(axis=1) / len(features)
        X_val[new_feature_3] = X_val[features].sub(X_val[features].mean(axis=1), axis=0).abs().sum(axis=1) / len(features)

        selected_features_df = X_train[features + [new_feature_1, new_feature_2, new_feature_3]]
        score = self.compute_total_correlation(selected_features_df)

        icr = sum(abs(X_train[feature].corr(y_train)) for feature in selected_features_df)
        icr = 1 / (icr + 1e-8)

        return score + icr

    def initialize_population(self, num_features):
        population = []
        for _ in range(self.population_size):
            chromosome = [0] * num_features
            selected_indices = random.sample(range(num_features), self.k)
            for idx in selected_indices:
                chromosome[idx] = 1
            population.append(chromosome)
        return population

    def binary_crossover(self, parent1, parent2):
        length = len(parent1)
        point1, point2 = sorted(random.sample(range(length), 2))

        def fix(chrom):
            ones = [i for i, b in enumerate(chrom) if b == 1]
            zeros = [i for i, b in enumerate(chrom) if b == 0]
            while len(ones) > self.k:
                i = random.choice(ones)
                chrom[i] = 0
                ones.remove(i)
            while len(ones) < self.k:
                i = random.choice(zeros)
                chrom[i] = 1
                zeros.remove(i)
                ones.append(i)
            return chrom

        child1 = fix(parent1[:point1] + parent2[point1:point2] + parent1[point2:])
        child2 = fix(parent2[:point1] + parent1[point1:point2] + parent2[point2:])
        return child1, child2

    def binary_mutation(self, chrom, rate=0.3):
        if random.random() < rate:
            ones = [i for i, b in enumerate(chrom) if b == 1]
            zeros = [i for i, b in enumerate(chrom) if b == 0]
            if ones and zeros:
                i, j = random.choice(ones), random.choice(zeros)
                chrom[i], chrom[j] = 0, 1
        return chrom

    def tournament_selection(self, population, scores):
        selected = []
        for _ in range(self.population_size // 2):
            candidates = random.sample(list(zip(scores, population)), self.tournament_size)
            selected.append(min(candidates, key=lambda x: x[0])[1])
        return selected

    def run(self):
        self.load_and_prepare_data()
        feature_names = self.feature_columns
        num_features = len(feature_names)
        population = self.initialize_population(num_features)

        for generation in range(self.generations):
            selected_features = [[feature_names[i] for i in range(num_features) if chrom[i] == 1] for chrom in population]
            fitness_scores = [self.fitness_function(features) for features in selected_features]

            best_idx = np.argmin(fitness_scores)
            best_features = selected_features[best_idx]

            print(f"Generation {generation+1}, Best Score: {fitness_scores[best_idx]}")

            X = self.scaled_features_df[best_features]
            y = self.scaled_features_df[self.target_column]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)

            self.scoresGA.append(auc)
            self.scoresACC.append(acc)
            self.scoresMI.append(fitness_scores[best_idx])

            parents = self.tournament_selection(population, fitness_scores)
            next_gen = []
            while len(next_gen) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = self.binary_crossover(p1, p2)
                next_gen.append(self.binary_mutation(c1))
                if len(next_gen) < self.population_size:
                    next_gen.append(self.binary_mutation(c2))
            population = next_gen

        return best_features

    def plot_metrics(self):
        generations = range(1, len(self.scoresGA) + 1)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        axes[0].plot(generations, self.scoresGA, marker='o', color='blue')
        axes[0].set_title("ROC AUC Score Over Generations")
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("ROC AUC")

        axes[1].plot(generations, self.scoresACC, marker='s', color='orange')
        axes[1].set_title("Accuracy Over Generations")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Accuracy")

        axes[2].plot(generations, self.scoresMI, marker='^', color='green')
        axes[2].set_title("Mutual Information + TC Fitness Over Generations")
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("Fitness Score")

        plt.tight_layout()
        plt.show()

def main():
    data_path = '/../home-credit-default-risk/application_train.csv'
    selector = GeneticFeatureSelector(data_path=data_path)
    best_features = selector.run()
    print("Best Feature Combination Found:", best_features)
    selector.plot_metrics()

if __name__ == "__main__":
    main()
