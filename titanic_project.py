import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import warnings

# Load and preprocess the Titanic dataset
df = pd.read_csv('titanic.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Prepare features and labels for the model
features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

labels = df['Survived']

def knn_impute(df):

    imputer = KNNImputer(n_neighbors=5)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# K-fold Cross-validation with different K for KNN
from sklearn.neighbors import KNeighborsClassifier

def k_fold_knn():
    k_range = range(1, 21)  # K values from 1 to 21
    mse_results = []
    rmse_results = []
    
    # K-fold Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_range:
        mse_fold = []
        for train_index, test_index in kf.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            
            # Impute missing values
            X_train_imputed = knn_impute(X_train)
            X_test_imputed = knn_impute(X_test)
            
            # Train KNN model
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train_imputed, y_train)
            y_pred = model.predict(X_test_imputed)
            
            # Calculate MSE and RMSE
            mse = mean_squared_error(y_test, y_pred)
            mse_fold.append(mse)
        
        mse_results.append(np.mean(mse_fold))
        rmse_results.append(np.sqrt(np.mean(mse_fold)))

    best_k = k_range[np.argmin(mse_results)]
    best_mse = mse_results[np.argmin(mse_results)]
    best_rmse = np.sqrt(best_mse)
    return k_range, mse_results, rmse_results, best_k, best_mse, best_rmse


# Function to display predictive model creation
def show_predictive_model_creation():
    st.subheader('Predictive Model Creation with KNN Imputation')
    st.write("""
    At the pre-processing part before building the predictive model, we compute these steps:
    - Filling Missing Age Values: 
    We replace missing values in the `Age` column with the median age of all passengers. The median is used to reduce the impact of outliers and to provide a better representation of the central tendency.

    - Dropping Rows with Missing Embarked Values: 
    We remove any rows where the `Embarked` column has missing values. This variable is crucial for our analysis, and retaining incomplete records would hinder our predictive modeling. We also notice there is less than %1 of missing values.

    - Encoding the Sex Variable: 
    The `Sex` variable is converted into a numerical format, with males mapped to `0` and females to `1`. This encoding allows us to use `Sex` as a predictor variable in machine learning models.

    - Omission of the Ticket Variable: 
    Regarding the presence of missing values in the `Cabin` variable, we have decided to omit it as a predictor variable.
    Due to the complexity of 'Ticket' varibale, we have decided to omit it as a predictor variable. This helps and ensures the integrity of the model.
    """)

    k_range, mse_results, rmse_results, best_k, best_mse, best_rmse = k_fold_knn()
    
    st.subheader('Best neighbour number on KNN')
    st.write(f"Best K: {best_k}")
    st.write(f"Best MSE: {best_mse:.4f}")
    st.write(f"Best RMSE: {best_rmse:.4f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, mse_results, marker='o', label='MSE', color='blue')
    plt.plot(k_range, rmse_results, marker='o', label='RMSE', color='orange')
    plt.title('MSE and RMSE vs K Value')
    plt.xlabel('K Value (Number of Neighbors)')
    plt.ylabel('Error')
    plt.xticks(k_range)
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Function to display variable explanations
def show_variable_explanations():
    st.subheader('Variable Explanations')
    variable_info = {
        'PassengerId': 'A unique identifier for each passenger.',
        'Survived': 'Indicates whether a passenger survived (1) or did not survive (0).',
        'Pclass': 'Passenger class: 1 = First class, 2 = Second class, 3 = Third class.',
        'Name': 'Full name of the passenger.',
        'Sex': 'Gender of the passenger (male or female).',
        'Age': 'Age of the passenger in years.',
        'SibSp': 'Number of siblings or spouses aboard.',
        'Parch': 'Number of parents or children aboard.',
        'Ticket': 'Ticket number of the passenger.',
        'Fare': 'Amount paid for the ticket.',
        'Cabin': 'Cabin number where the passenger stayed.',
        'Embarked': 'Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton.'
    }
    for variable, description in variable_info.items():
        st.write(f"**{variable}:** {description}")

# Function to display distributions
import matplotlib.pyplot as plt
import seaborn as sns

def show_distributions():
    st.subheader('Distributions of Key Variables')

    # List of variables to visualize
    variables = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']

    for var in variables:
        # Histogram with KDE
        plt.figure(figsize=(10, 5))
        sns.histplot(df[var].dropna(), bins=30, kde=True)
        plt.title(f'{var} Distribution of Passengers')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Boxplot for outliers
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[var])
        plt.title(f'{var} Boxplot')
        plt.xlabel(var)
        st.pyplot(plt)

    # Additional variable for categorical data
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Pclass', data=df)
    plt.title('Passenger Class Distribution')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Distribution of Sex
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Sex', data=df)
    plt.title('Gender Distribution of Passengers')
    plt.xlabel('Sex')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Distribution of Survival
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Distribution')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    st.pyplot(plt)


# Function to display missing values
def show_missing_values():
    st.subheader('Missing Values in the Dataset')
    df = pd.read_csv('titanic.csv')
    # Calculate the total number of entries
    total_entries = df.shape[0]
    
    # Calculate missing values and the percentage of missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / total_entries) * 100
    
    # Create a DataFrame for better display
    missing_data = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })
    
    
    # Reset index for better presentation
    missing_data.reset_index(inplace=True)
    missing_data.rename(columns={'index': 'Column'}, inplace=True)
    
    # Display the DataFrame
    st.write(missing_data)


# Function to display correlation matrix
def show_correlation_matrix():
    st.subheader('Correlation Matrix')

    correlation_matrix = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()

    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True, square=True)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)

# Function to predict survival
def predict_survival():
    st.subheader('Predict Survival using KNN')
    
    # Determine the best k from K-fold cross-validation
    _, _, _, best_k, _, _ = k_fold_knn()
    
    # Input fields for user data
    pclass = st.selectbox('Passenger Class', (1, 2, 3))
    sex = st.selectbox('Sex', ('male', 'female'))
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, value=0)
    parch = st.number_input('Number of Parents/Children Aboard', min_value=0, value=0)
    fare = st.number_input('Fare', min_value=0.0, value=50.0)

    # Map the sex input to numerical
    sex_encoded = 0 if sex == 'male' else 1
    
    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })

    # Impute missing values for input data
    input_data_imputed = knn_impute(input_data)

    # Predict survival using KNN with the best k
    if st.button('Predict'):
        model = KNeighborsClassifier(n_neighbors=best_k)
        # Combine original features and labels to retrain the model
        model.fit(features, labels)
        prediction = model.predict(input_data_imputed)
        st.write(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Not Survived'}")

import matplotlib.pyplot as plt

def show_survival_by_class():
    st.subheader('Percentage of Survivors by Class')

    # Calcular el porcentaje de supervivencia por clase
    survival_counts = df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack().fillna(0)

    plt.figure(figsize=(8, 8))
    for pclass in survival_counts.index:
        plt.subplot(3,1, pclass)
        survival_counts.loc[pclass].plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107'], labels=['No Survived', 'Survived'])
        plt.title(f'{pclass} Class')
        plt.ylabel('')

    plt.tight_layout()
    st.pyplot(plt)


st.title('Titanic Data Analysis Dashboard')

st.sidebar.title('Navigation')
option = st.sidebar.radio("Select an option:", 
                           ('Variable Explanations', 
                            'Distributions', 
                            'Missing Values', 
                            'Correlation Matrix', 
                            'Predict Survival', 
                            'Predictive Model Creation',
                            'Research Questions'))

# Show the selected section
if option == 'Variable Explanations':
    show_variable_explanations()
elif option == 'Distributions':
    show_distributions()
elif option == 'Missing Values':
    show_missing_values()
elif option == 'Correlation Matrix':
    show_correlation_matrix()
elif option == 'Predict Survival':
    predict_survival()
elif option == 'Predictive Model Creation':
    show_predictive_model_creation()
elif option == 'Research Questions':
    show_survival_by_class()

# Run the Streamlit app
if __name__ == '__main__':
    st.write("")
