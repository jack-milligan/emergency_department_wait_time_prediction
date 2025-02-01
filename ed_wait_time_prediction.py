import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def simulate_synthetic_data(num_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulate synthetic emergency department (ED) data.

    Parameters:
        num_samples: Number of data samples to generate.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame with simulated ED data.
    """
    np.random.seed(random_seed)

    data = {
        'arrival_time': np.random.randint(0, 24, num_samples),  # Hour of arrival (0-23)
        'acuity_level': np.random.choice(
            [1, 2, 3, 4, 5],
            num_samples,
            p=[0.1, 0.2, 0.3, 0.2, 0.2]
        ),
        'current_patient_volume': np.random.randint(10, 100, num_samples),  # Current number of patients in ED
        'staff_on_duty': np.random.randint(5, 20, num_samples)  # Number of staff available
    }

    df = pd.DataFrame(data)
    return df


def simulate_wait_time(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulate wait time based on ED features and add a 'wait_time' column.

    Wait time is modeled as:
        30 + 5 * (patient_volume/staff_on_duty) - 10 * (1/acuity_level) + random noise

    Parameters:
        df: DataFrame containing ED features.
        random_seed: Seed for noise reproducibility.

    Returns:
        DataFrame with the simulated 'wait_time' column.
    """
    num_samples = len(df)
    np.random.seed(random_seed)

    df['wait_time'] = (
            30 +
            5 * df['current_patient_volume'] / df['staff_on_duty'] -
            10 * (1 / df['acuity_level']) +
            np.random.normal(0, 5, num_samples)
    )
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the DataFrame into training and testing sets.

    Parameters:
        df: DataFrame containing the data.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    features = df[['arrival_time', 'acuity_level', 'current_patient_volume', 'staff_on_duty']]
    target = df['wait_time']
    return train_test_split(features, target, test_size=test_size, random_state=random_state)


def build_and_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Build and train a linear regression model.

    Parameters:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> (np.ndarray, float, float):
    """
    Predict wait times and evaluate the model using MAE and R².

    Parameters:
        model: Trained LinearRegression model.
        X_test: Testing features.
        y_test: Testing target.

    Returns:
        y_pred: Predicted wait times.
        mae: Mean Absolute Error.
        r2: R² score.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, r2


def print_evaluation(mae: float, r2: float) -> None:
    """
    Print evaluation metrics.

    Parameters:
        mae: Mean Absolute Error.
        r2: R² score.
    """
    print(f"MAE: {mae:.2f}, R²: {r2:.2f}")


def save_and_show_plot(y_test: pd.Series, y_pred: np.ndarray, image_path: str = "images/scatter_plot.png") -> None:
    """
    Create a scatter plot of actual vs. predicted wait times, save it, and display the plot.

    Parameters:
        y_test: Actual wait times.
        y_pred: Predicted wait times.
        image_path: File path to save the plot.
    """
    # Ensure the images folder exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Prediction')
    plt.xlabel("Actual Wait Time")
    plt.ylabel("Predicted Wait Time")
    plt.title("Actual vs Predicted Wait Times")
    plt.legend()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main() -> None:
    # Step 1: Simulate synthetic ED data
    df = simulate_synthetic_data()

    # Step 2: Simulate wait times and add to the DataFrame
    df = simulate_wait_time(df)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Step 4: Build and train the linear regression model
    model = build_and_train_model(X_train, y_train)

    # Step 5: Make predictions and evaluate the model
    y_pred, mae, r2 = evaluate_model(model, X_test, y_test)
    print_evaluation(mae, r2)

    # Step 6: Visualize the actual vs. predicted wait times
    save_and_show_plot(y_test, y_pred)


if __name__ == "__main__":
    main()
