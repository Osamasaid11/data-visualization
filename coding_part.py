import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
googleplaystore_df = pd.read_csv('googleplaystore.csv')
googleplaystore_user_reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')

# Data Cleaning and Preparation
googleplaystore_df.drop(index=googleplaystore_df[googleplaystore_df['Category'] == '1.9'].index, inplace=True)  # Removing problematic row based on category '1.9'
googleplaystore_df['Last Updated'] = pd.to_datetime(googleplaystore_df['Last Updated'], errors='coerce')
googleplaystore_df['Installs'] = googleplaystore_df['Installs'].str.replace('+', '').str.replace(',', '').astype(int)
googleplaystore_df['Reviews'] = pd.to_numeric(googleplaystore_df['Reviews'], errors='coerce')
googleplaystore_user_reviews_df.dropna(subset=['Translated_Review', 'Sentiment'], inplace=True)

# Descriptive Statistics, Correlation, Skewness, and Kurtosis
descriptive_stats = googleplaystore_df.describe()
correlation_matrix = googleplaystore_df[['Rating', 'Reviews', 'Installs']].corr()
skewness = googleplaystore_df[['Rating', 'Reviews', 'Installs']].skew()
kurtosis = googleplaystore_df[['Rating', 'Reviews', 'Installs']].kurtosis()

# Function for Histogram of App Ratings
def plot_histogram_app_ratings():
    plt.figure(figsize=(10, 6))
    sns.histplot(googleplaystore_df['Rating'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of App Ratings on Google Play Store')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# Function for Bar Chart of Apps per Category
def plot_bar_apps_per_category():
    plt.figure(figsize=(12, 8))
    category_counts = googleplaystore_df['Category'].value_counts()
    sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
    plt.title('Count of Apps per Category on Google Play Store')
    plt.xlabel('Count of Apps')
    plt.ylabel('Category')
    plt.show()

# Function for Pie Chart of Sentiments Distribution
def plot_pie_sentiments_distribution():
    sentiment_distribution = googleplaystore_user_reviews_df['Sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    sentiment_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'salmon', 'lightblue'], explode=(0.1, 0, 0), shadow=True)
    plt.title('Distribution of Sentiments from User Reviews')
    plt.ylabel('')
    plt.show()

# Function for Line Chart of Apps Added Over Time
def plot_line_apps_over_time():
    apps_per_year = googleplaystore_df.groupby(googleplaystore_df['Last Updated'].dt.year).size()
    plt.figure(figsize=(12, 6))
    apps_per_year.plot(kind='line', marker='o', color='teal')
    plt.title('Trends in the Number of Apps Added Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Apps Added')
    plt.grid(True)
    plt.show()

# Function for Heatmap of Correlation Matrix
def plot_heatmap_correlation():
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix Among Numeric Variables')
    plt.show()

# Execute the functions to generate the visualizations
plot_histogram_app_ratings()
plot_bar_apps_per_category()
plot_pie_sentiments_distribution()
plot_line_apps_over_time()
plot_heatmap_correlation()

# Print the descriptive statistics, correlation, skewness, and kurtosis
print("Descriptive Statistics:\n", descriptive_stats)
print("\nCorrelation Matrix:\n", correlation_matrix)
print("\nSkewness:\n", skewness)
print("\nKurtosis:\n", kurtosis)
