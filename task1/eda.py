# Exploratory Data Analysis (EDA)
sns.pairplot(df, hue='species')
plt.show()

# Check for class distribution
print(df['species'].value_counts())

# Describe the dataset
print(df.describe())