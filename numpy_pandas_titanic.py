# ============================================================
# NumPy, Pandas and Titanic Data Analysis
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# PART 1 - NumPy
# ============================================================

print("=" * 70)
print("PART 1 - NumPy")
print("=" * 70)

# Exercise 1: Create 20 random sales values
np.random.seed(42)

sales_20_days = np.random.randint(100, 1001, size=20)

print("\n1. Sales for 20 consecutive days:")
print(sales_20_days)

# Reshape into 4 x 5
sales_4x5 = sales_20_days.reshape(4, 5)

print("\nSales reshaped into 4 x 5:")
print(sales_4x5)


# Exercise 2: Compare two branches
branch_1 = np.array([500, 650, 720, 580, 900])
branch_2 = np.array([450, 700, 680, 620, 850])

total_sales = branch_1 + branch_2
sales_difference = branch_1 - branch_2

print("\n2. Branch 1 sales:")
print(branch_1)

print("\nBranch 2 sales:")
print(branch_2)

print("\nTotal sales of both branches:")
print(total_sales)

print("\nDaily difference (Branch 1 - Branch 2):")
print(sales_difference)


# Exercise 3: NumPy statistics
numpy_mean = np.mean(sales_20_days)
numpy_std = np.std(sales_20_days)
numpy_max = np.max(sales_20_days)
numpy_min = np.min(sales_20_days)

print("\n3. NumPy Statistics:")
print("Mean:", numpy_mean)
print("Standard Deviation:", numpy_std)
print("Maximum:", numpy_max)
print("Minimum:", numpy_min)


# Manual functions from Task 2
def manual_mean(data):
    return sum(data) / len(data)


def manual_variance(data):
    mean = manual_mean(data)
    total = 0

    for value in data:
        total += (value - mean) ** 2

    return total / len(data)


def manual_std(data):
    return manual_variance(data) ** 0.5


def manual_max(data):
    maximum = data[0]

    for value in data:
        if value > maximum:
            maximum = value

    return maximum


def manual_min(data):
    minimum = data[0]

    for value in data:
        if value < minimum:
            minimum = value

    return minimum


manual_mean_result = manual_mean(sales_20_days)
manual_std_result = manual_std(sales_20_days)
manual_max_result = manual_max(sales_20_days)
manual_min_result = manual_min(sales_20_days)

print("\nComparison between NumPy and Manual Functions:")

print("Mean - NumPy:", numpy_mean)
print("Mean - Manual:", manual_mean_result)

print("Standard Deviation - NumPy:", numpy_std)
print("Standard Deviation - Manual:", manual_std_result)

print("Maximum - NumPy:", numpy_max)
print("Maximum - Manual:", manual_max_result)

print("Minimum - NumPy:", numpy_min)
print("Minimum - Manual:", manual_min_result)

print("\nResults match:")

print("Mean:", np.isclose(numpy_mean, manual_mean_result))
print("Standard Deviation:", np.isclose(numpy_std, manual_std_result))
print("Maximum:", numpy_max == manual_max_result)
print("Minimum:", numpy_min == manual_min_result)


# Exercise 4: Slicing
print("\n4. Slicing")

week_2 = sales_4x5[1]

print("\nWeek 2:")
print(week_2)

day_3 = sales_4x5[:, 2]

print("\nDay 3 across all weeks:")
print(day_3)

sub_matrix = sales_4x5[0:2, 1:4]

print("\nSub-matrix:")
print(sub_matrix)


# ============================================================
# PART 2 - Pandas: Series and DataFrame
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - Pandas")
print("=" * 70)


# Exercise 1: Series
sales_series = pd.Series(
    [500, 650, 720, 580, 900],
    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
)

print("\n1. Sales Series:")
print(sales_series)

print("\nSeries Index:")
print(sales_series.index)

print("\nSeries Values:")
print(sales_series.values)

print("\nSeries Data Type:")
print(sales_series.dtype)


# Exercise 2: DataFrame from Dictionary
products = {
    "Product": ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"],
    "Price": [800, 500, 300, 250, 50],
    "Quantity_Sold": [10, 20, 15, 12, 30]
}

products_df = pd.DataFrame(products)

print("\n2. Products DataFrame:")
print(products_df)


# Exercise 3: Titanic dataset
titanic = sns.load_dataset("titanic")

print("\n3. Titanic - First 5 rows:")
print(titanic.head())

print("\nShape:")
print(titanic.shape)

print("\nColumns:")
print(titanic.columns)

print("\nData Types:")
print(titanic.dtypes)

print("\nStatistical Description:")
print(titanic.describe())

print("\nMissing Values:")
print(titanic.isnull().sum())


# ============================================================
# PART 3 - Selecting and Filtering Data
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - Selecting and Filtering")
print("=" * 70)


# Exercise 1: Age column
print("\n1. Age column:")
print(titanic["age"])


# Exercise 2: Passengers older than 40
older_than_40 = titanic[titanic["age"] > 40]

print("\n2. Passengers older than 40:")
print(older_than_40)


# Exercise 3: Female passengers who survived
female_survivors = titanic[
    (titanic["sex"] == "female") &
    (titanic["survived"] == 1)
]

print("\n3. Female passengers who survived:")
print(female_survivors)


# Exercise 4: Top 10 passengers by fare
top_10_fares = titanic.sort_values(
    by="fare",
    ascending=False
).head(10)

print("\n4. Top 10 passengers by fare:")
print(top_10_fares)


# Exercise 5: loc
selected_data = titanic.loc[5:10][["age", "pclass"]]

print("\n5. Rows 5 to 10 with selected columns:")
print(selected_data)


# ============================================================
# PART 4 - Titanic Initial Data Exploration Report
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - Titanic Initial Data Exploration Report")
print("=" * 70)


# ------------------------------------------------------------
# 1. General Overview
# ------------------------------------------------------------

print("\n1. GENERAL OVERVIEW")

print("Number of rows:", titanic.shape[0])
print("Number of columns:", titanic.shape[1])

print("\nColumn names:")
print(titanic.columns.tolist())


# ------------------------------------------------------------
# Column descriptions
# ------------------------------------------------------------

print("\nColumn Descriptions:")

print("""
survived     = Whether the passenger survived (1 = Yes, 0 = No)
pclass       = Passenger class (1 = First, 2 = Second, 3 = Third)
sex          = Passenger gender
age          = Passenger age
sibsp        = Number of siblings/spouses aboard
parch        = Number of parents/children aboard
fare         = Passenger fare
embarked     = Port where the passenger boarded
class        = Passenger class as text
who          = Passenger category (man, woman, child)
adult_male   = Whether the passenger was an adult male
deck         = Deck location
embark_town  = Port town where passenger boarded
alive        = Survival status as text
alone        = Whether passenger was traveling alone
""")


# ------------------------------------------------------------
# 2. Data Quality
# ------------------------------------------------------------

print("\n2. DATA QUALITY - MISSING VALUES")

missing_values = titanic.isnull().sum()

print("\nMissing values by column:")
print(missing_values)

missing_percentage = (titanic.isnull().sum() / len(titanic)) * 100

print("\nPercentage of missing values by column:")
print(missing_percentage.round(2))

print("\nColumns with a high percentage of missing data:")

for column in missing_percentage.index:
    if missing_percentage[column] > 50:
        print(
            f"{column}: {missing_percentage[column]:.2f}% missing"
        )


# ------------------------------------------------------------
# 3. Descriptive Statistics
# ------------------------------------------------------------

print("\n3. DESCRIPTIVE STATISTICS")

print(titanic.describe())

print("\nImportant observations:")

average_fare = titanic["fare"].mean()
minimum_age = titanic["age"].min()
maximum_age = titanic["age"].max()

print(f"Average ticket fare: {average_fare:.2f}")
print(f"Minimum recorded age: {minimum_age:.2f}")
print(f"Maximum recorded age: {maximum_age:.0f}")


# ------------------------------------------------------------
# 4. Business Question 1
# ------------------------------------------------------------

print("\n4. BUSINESS QUESTIONS")

survival_rate = titanic["survived"].mean() * 100

print("\nQ1: What percentage of passengers survived?")
print(f"Survival rate: {survival_rate:.2f}%")

print(
    f"Answer: Approximately {survival_rate:.2f}% "
    "of the passengers survived."
)


# ------------------------------------------------------------
# 5. Business Question 2
# ------------------------------------------------------------

survival_by_sex = titanic.groupby("sex")["survived"].mean() * 100

print("\nQ2: Is the survival rate among females higher than males?")
print(survival_by_sex.round(2))

female_survival = survival_by_sex["female"]
male_survival = survival_by_sex["male"]

if female_survival > male_survival:
    print(
        f"Answer: Yes. Female survival was {female_survival:.2f}%, "
        f"compared with {male_survival:.2f}% for males."
    )
else:
    print("Answer: No.")


# ------------------------------------------------------------
# 6. Business Question 3
# ------------------------------------------------------------

survival_by_class = titanic.groupby("pclass")["survived"].mean() * 100

print("\nQ3: Did first-class passengers have a higher survival rate")
print("than third-class passengers?")

print("\nSurvival rate by passenger class:")
print(survival_by_class.round(2))

first_class_survival = survival_by_class[1]
third_class_survival = survival_by_class[3]

if first_class_survival > third_class_survival:
    print(
        f"Answer: Yes. First-class survival was "
        f"{first_class_survival:.2f}%, compared with "
        f"{third_class_survival:.2f}% for third-class passengers."
    )
else:
    print("Answer: No.")


# ------------------------------------------------------------
# 7. Business Question 4
# ------------------------------------------------------------

average_age_by_survival = titanic.groupby("survived")["age"].mean()

survivors_average_age = average_age_by_survival[1]
non_survivors_average_age = average_age_by_survival[0]

print("\nQ4: What was the average age of survivors")
print("compared with non-survivors?")

print(
    f"Average age of survivors: "
    f"{survivors_average_age:.2f} years"
)

print(
    f"Average age of non-survivors: "
    f"{non_survivors_average_age:.2f} years"
)

print(
    f"Answer: Survivors were approximately "
    f"{survivors_average_age:.2f} years old on average, "
    f"while non-survivors were approximately "
    f"{non_survivors_average_age:.2f} years old."
)


# ============================================================
# FINAL BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS SUMMARY")
print("=" * 70)

print("""
The Titanic dataset provides several important business insights.
First, the overall survival rate was relatively low, with approximately
38% of passengers surviving. Second, female passengers had a much higher
survival rate than male passengers. Third, passenger class was strongly
associated with survival, as first-class passengers had a substantially
higher survival rate than third-class passengers. The dataset also has
data quality limitations, particularly the deck column, which contains
a large proportion of missing values. These findings show why both
data quality and segmentation are important when analyzing historical
data and making business decisions.
""")

print("=" * 70)
print("END OF ANALYSIS")
print("=" * 70)