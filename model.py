import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression

columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
data = pd.read_csv("data/abalone.data", names=columns)

data['Sex'] = data['Sex'].map({'M': 1, 'F': 0, 'I': 2})


# Correlation heatmap
corr_matrix = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('plot_correlation_heatmap.png')
plt.close()

# Scatter plots
positive_corr_feature = 'Shell_weight'
negative_corr_feature = 'Sex'
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[positive_corr_feature], data['Rings'], alpha=0.5)
plt.title(f'Scatter Plot: {positive_corr_feature} vs Rings')
plt.xlabel(positive_corr_feature)
plt.ylabel('Rings')
plt.subplot(1, 2, 2)
plt.scatter(data[negative_corr_feature], data['Rings'], alpha=0.5)
plt.title(f'Scatter Plot: {negative_corr_feature} vs Rings')
plt.xlabel(negative_corr_feature)
plt.ylabel('Rings')
plt.savefig('plot_scatter_plots.png')
plt.close()

# Histograms
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.hist(data[positive_corr_feature], bins=30, alpha=0.7, color='blue')
plt.title(f'Histogram of {positive_corr_feature}')
plt.subplot(1, 3, 2)
plt.hist(data[negative_corr_feature], bins=30, alpha=0.7, color='green')
plt.title(f'Histogram of {negative_corr_feature}')
plt.subplot(1, 3, 3)
plt.hist(data['Rings'], bins=30, alpha=0.7, color='red')
plt.title('Histogram of Rings')
plt.savefig('plot_histograms.png')
plt.close()


def split_data(X, y, experiment_number):
    random_seed = experiment_number
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_seed
    )
    return X_train, X_test, y_train, y_test

# Q1 Classification
X = data.drop(columns='Rings')
y = data['Rings']
y_classification = (data['Rings'] > 7).astype(int)
num_experiments = 3
init_number = 122
acc_list = []
auc_list = []

for exp_num in range(num_experiments):
    X_train, X_test, y_train, y_test = split_data(X, y_classification, init_number + exp_num)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_class = np.heaviside(y_pred, 1)
    accuracy = accuracy_score(y_test, y_pred_class)
    acc_list.append(accuracy)
    auc = roc_auc_score(y_test, y_pred)
    auc_list.append(auc)

acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
print(f"Results over {num_experiments} experiments (Linear Regression):")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"AUC Score: {auc_mean:.4f} ± {auc_std:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc_list[-1]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('plot_roc_curve.png')
plt.close()

# Q2 Linear Regression with/without normalization
def run_linear_regression_experiments(X, y, num_experiments=3, init_number=222, use_normalization=False):
    train_rmse_list, test_rmse_list = [], []
    train_r2_list, test_r2_list = [], []
    y_test_final, y_pred_test_final = None, None  # To store for final plot
    
    for exp_num in range(num_experiments):
        X_train, X_test, y_train, y_test = split_data(X, y, init_number + exp_num)
        
        if use_normalization:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        train_rmse_list.append(rmse_train)
        test_rmse_list.append(rmse_test)
        train_r2_list.append(r2_train)
        test_r2_list.append(r2_test)
        
        # Store the last experiment's test results for plotting
        if exp_num == num_experiments - 1:
            y_test_final = y_test
            y_pred_test_final = y_pred_test

    # Plotting the last experiment's test results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test_final, y_test_final, color='blue', label='Predictions', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Ideal Line')
    plt.xlabel('Predicted Ring Age')
    plt.ylabel('Actual Ring Age')
    plt.title(f'Linear Regression Model: Predicted vs Actual Ring Age {"normalized" if use_normalization else "not_normalized"}')
    plt.legend()
    plt.savefig(f'plot_linear_regression_Predicted vs Actual Ring Age_{"normalized" if use_normalization else "not_normalized"}.png')
    plt.close()

    # Return results
    results = {
        "Training RMSE": (np.mean(train_rmse_list), np.std(train_rmse_list)),
        "Testing RMSE": (np.mean(test_rmse_list), np.std(test_rmse_list)),
        "Training R-squared": (np.mean(train_r2_list), np.std(train_r2_list)),
        "Testing R-squared": (np.mean(test_r2_list), np.std(test_r2_list))
    }
    return results

X = data.drop('Rings', axis=1)
y = data['Rings']

print("Linear Regression with normalization:")
results = run_linear_regression_experiments(X, y, use_normalization=True)
for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

print("\nLinear Regression without normalization:")
results = run_linear_regression_experiments(X, y, use_normalization=False)
for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

# Q3 Linear Regression with/without normalization (selected features)
X = data[['Shell_weight', 'Diameter']]
y = data['Rings']

print("\nLinear Regression with normalization (selected features):")
results = run_linear_regression_experiments(X, y, use_normalization=True)
for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

print("\nLinear Regression without normalization (selected features):")
results = run_linear_regression_experiments(X, y, use_normalization=False)
for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.4f} ± {std:.4f}")

# Q3 Logistic Regression with/without normalization
def run_logistic_regression_experiments(X, y, num_experiments=3, init_number=2232, use_normalization=False):
    train_acc_list, test_acc_list = [], []
    train_f1_list, test_f1_list = [], []
    train_auc_list, test_auc_list = [], []
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    
    for exp_num in range(num_experiments):
        X_train, X_test, y_train, y_test = split_data(X, y, init_number + exp_num)
        
        if use_normalization:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred_prob_test = model.predict_proba(X_test)[:, 1]
        
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
        auc_train = roc_auc_score(y_train, y_pred_prob_train)
        auc_test = roc_auc_score(y_test, y_pred_prob_test)
        
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)
        train_f1_list.append(f1_train)
        test_f1_list.append(f1_test)
        train_auc_list.append(auc_train)
        test_auc_list.append(auc_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
        tpr_list.append(np.interp(mean_fpr, fpr, tpr))
        tpr_list[-1][0] = 0.0
    
    train_acc_mean, train_acc_std = np.mean(train_acc_list), np.std(train_acc_list)
    test_acc_mean, test_acc_std = np.mean(test_acc_list), np.std(test_acc_list)
    train_f1_mean, train_f1_std = np.mean(train_f1_list), np.std(train_f1_list)
    test_f1_mean, test_f1_std = np.mean(test_f1_list), np.std(test_f1_list)
    train_auc_mean, train_auc_std = np.mean(train_auc_list), np.std(train_auc_list)
    test_auc_mean, test_auc_std = np.mean(test_auc_list), np.std(test_auc_list)
    
    print(f"Results over {num_experiments} experiments (Logistic Regression):")
    print(f"Normalization: {'Applied' if use_normalization else 'Not Applied'}")
    print(f"Training Accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
    print(f"Testing Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    print(f"Training F1 Score: {train_f1_mean:.4f} ± {train_f1_std:.4f}")
    print(f"Testing F1 Score: {test_f1_mean:.4f} ± {test_f1_std:.4f}")
    print(f"Training AUC Score: {train_auc_mean:.4f} ± {train_auc_std:.4f}")
    print(f"Testing AUC Score: {test_auc_mean:.4f} ± {test_auc_std:.4f}")
    
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(test_auc_list)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Normalization: {"Applied" if use_normalization else "Not Applied"})')
    plt.legend(loc="lower right")
    plt.savefig(f'plot_logistic_regression_roc_{"normalized" if use_normalization else "not_normalized"}.png')
    plt.close()

    return {
        'train_acc': (train_acc_mean, train_acc_std),
        'test_acc': (test_acc_mean, test_acc_std),
        'train_f1': (train_f1_mean, train_f1_std),
        'test_f1': (test_f1_mean, test_f1_std),
        'train_auc': (train_auc_mean, train_auc_std),
        'test_auc': (test_auc_mean, test_auc_std)
    }
#Q3
y = (data['Rings'] > 7).astype(int)
X = data[['Shell_weight', 'Diameter']]

print("\nLogistic Regression without normalization:")
results_without_norm = run_logistic_regression_experiments(X, y, use_normalization=False)

print("\nLogistic Regression with normalization:")
results_with_norm = run_logistic_regression_experiments(X, y, use_normalization=True)


#Q4 Regression

#Q4 with/without normalization



def run_neural_network_experiments(X, y, use_normalization=False):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4456)
    
    if use_normalization:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    def create_model(hidden_layers, neurons, learning_rate):
        model = Sequential()
        model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))
        
        for _ in range(hidden_layers - 1):
            model.add(Dense(neurons, activation='relu'))
        
        model.add(Dense(1))
        
        optimizer = SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    # Hyperparameter search
    hidden_layers_options = [1]
    neurons_options = [8]
    learning_rate_options = [0.01]
    results = []
    
    for hidden_layers in hidden_layers_options:
        for neurons in neurons_options:
            for lr in learning_rate_options:
                model = create_model(hidden_layers, neurons, lr)
                history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
                
                y_pred = model.predict(X_test_scaled).flatten()
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'hidden_layers': hidden_layers,
                    'neurons': neurons,
                    'learning_rate': lr,
                    'mse': mse,
                    'r2': r2
                })
                
                print(f"Layers: {hidden_layers}, Neurons: {neurons}, LR: {lr}, MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Find the best model
    best_model = min(results, key=lambda x: x['mse'])
    print("\nBest Model:")
    print(f"Hidden Layers: {best_model['hidden_layers']}")
    print(f"Neurons: {best_model['neurons']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"MSE: {best_model['mse']:.4f}")
    print(f"R2: {best_model['r2']:.4f}")
    
    # Train the best model
    best_nn = create_model(best_model['hidden_layers'], best_model['neurons'], best_model['learning_rate'])
    history = best_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Training History (Normalization: {"Applied" if use_normalization else "Not Applied"})')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(f'Model Training History_{"normalized" if use_normalization else "not_normalized"}.png')
    plt.close()
    
    return best_model, history

def run_trial_experiments(X, y, use_normalization=False):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4456)
    
    def create_classification_model(hidden_layers, neurons, learning_rate):
        model = Sequential()
        model.add(Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)))

        for _ in range(hidden_layers - 1):  # Add additional hidden layers
            model.add(Dense(neurons, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        optimizer = SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    if use_normalization:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Hyperparameter options
    hidden_layers_options = [1]
    neurons_options = [8]
    learning_rate_options = [0.01]
    
    results = []
    
    # Perform trial experiments with different combinations
    for hidden_layers in hidden_layers_options:
        for neurons in neurons_options:
            for lr in learning_rate_options:
                model = create_classification_model(hidden_layers, neurons, lr)
                history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
                
                y_pred = model.predict(X_test_scaled).flatten()
                y_pred_class = (y_pred > 0.5).astype(int)
                
                accuracy = accuracy_score(y_test, y_pred_class)
                auc = roc_auc_score(y_test, y_pred)
                
                results.append({
                    'hidden_layers': hidden_layers,
                    'neurons': neurons,
                    'learning_rate': lr,
                    'accuracy': accuracy,
                    'auc': auc
                })
                
                print(f"Layers: {hidden_layers}, Neurons: {neurons}, LR: {lr}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Find the best model based on AUC
    best_model = max(results, key=lambda x: x['auc'])
    print("\nBest Model:")
    print(f"Hidden Layers: {best_model['hidden_layers']}")
    print(f"Neurons: {best_model['neurons']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"AUC: {best_model['auc']:.4f}")
    
    # Plot ROC curve for the best model
    best_nn = create_classification_model(best_model['hidden_layers'], best_model['neurons'], best_model['learning_rate'])
    best_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred = best_nn.predict(X_test_scaled).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {best_model["auc"]:.4f})',color='b')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Neural_Network_classification_{"normalized" if use_normalization else "not_normalized"}')
    plt.legend()
    plt.savefig(f'Neural_Network_classification_{"normalized" if use_normalization else "not_normalized"}.png')
    plt.close()
    
    return best_model, best_nn



X = data.drop('Rings', axis=1)
y = data['Rings']

# Run experiments without normalization
print("Running Neural Network experiments without normalization:")
best_model_without_norm, history_without_norm = run_neural_network_experiments(X, y, use_normalization=False)

# Run experiments with normalization
print("\nRunning Neural Network experiments with normalization:")
best_model_with_norm, history_with_norm = run_neural_network_experiments(X, y, use_normalization=True)

# Compare results
print("\nComparison of best models:")
print("Without normalization:")
print(f"MSE: {best_model_without_norm['mse']:.4f}")
print(f"R2: {best_model_without_norm['r2']:.4f}")
print("\nWith normalization:")
print(f"MSE: {best_model_with_norm['mse']:.4f}")
print(f"R2: {best_model_with_norm['r2']:.4f}")



y = (data['Rings'] > 7).astype(int)
X = data[['Shell_weight', 'Diameter']]


# Classification task 
# Run trial experiments without normalization
print("Running Trial Experiments for Binary Classification (Without Normalization):")
best_model_without_norm, best_nn_without_norm = run_trial_experiments(X, y, use_normalization=False)

# Run trial experiments with normalization
print("\nRunning Trial Experiments for Binary Classification (With Normalization):")
best_model_with_norm, best_nn_with_norm = run_trial_experiments(X, y, use_normalization=True)


