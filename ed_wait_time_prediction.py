# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LinearRegression  # For building a linear regression model
from sklearn.metrics import mean_absolute_error, r2_score  # For model evaluation metrics
import matplotlib.pyplot as plt  # For data visualization

# --- Step 1: Simulate synthetic data ---
# Set a random seed for reproducibility of the simulation
np.random.seed(42)

# Define the number of samples (rows) in the dataset
num_samples = 10000

# Create a dictionary to simulate emergency department (ED) data
ed_data = {
    'arrival_time': np.random.randint(0, 24, num_samples),  # Random hour of the day (0-23)
    'acuity_level': np.random.choice(  # Random patient acuity level (severity of condition)
        [1, 2, 3, 4, 5],  # Acuity levels (1 = critical, 5 = minor)
        num_samples,
        p=[0.1, 0.2, 0.3, 0.2, 0.2]  # Probability distribution for acuity levels
    ),
    'current_patient_volume': np.random.randint(10, 100, num_samples),  # Current number of patients in ED
    'staff_on_duty': np.random.randint(5, 20, num_samples),  # Number of staff currently available
    'wait_time': None  # Placeholder for wait time, which will be calculated later
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(ed_data)

# --- Step 2: Simulate wait time based on given features ---
# The formula for wait time combines various factors:
# - Base wait time of 30 minutes.
# - An increase proportional to patient volume and inversely proportional to staff availability.
# - A decrease based on higher acuity levels (more severe cases get faster attention).
# - Random noise is added to make the data more realistic.

df['wait_time'] = (
        30 + 5 * df['current_patient_volume'] / df['staff_on_duty']  # Higher patient volume increases wait time
        - 10 * (1 / df['acuity_level'])  # Higher acuity levels (lower numbers) reduce wait time
        + np.random.normal(0, 5, num_samples)  # Add random noise with mean 0 and standard deviation 5
)

# --- Step 3: Split data into training and testing sets ---
# Separate independent variables (features) and the dependent variable (target)
X = df[['arrival_time', 'acuity_level', 'current_patient_volume', 'staff_on_duty']]  # Features
y = df['wait_time']  # Target variable (wait time)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Build a linear regression model ---
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# --- Step 5: Make predictions on the testing set ---
# Use the trained model to predict wait times for the testing data
y_pred = model.predict(X_test)

# --- Step 6: Evaluate the model ---
# Calculate the Mean Absolute Error (MAE) - average magnitude of prediction errors
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R² score - proportion of variance explained by the model
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"MAE: {mae:.2f}, R²: {r2:.2f}")

# --- Step 7: Visualize the results ---
# Scatter plot of actual vs. predicted wait times
plt.scatter(y_test, y_pred, alpha=0.5)  # Add transparency for better visibility of overlapping points
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')  # Line indicating perfect prediction
plt.xlabel("Actual Wait Time")  # Label for x-axis
plt.ylabel("Predicted Wait Time")  # Label for y-axis
plt.title("Actual vs Predicted Wait Times")  # Title of the plot
plt.show()  # Display the plot
