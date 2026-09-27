# WEEK 2 FINAL PROJECT
# Titanic: EDA -> Cleaning -> Merging -> Advanced Visualization
# Tasks 3, 4 and 5 combined

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ============================================================
# STEP 1 - CREATE THE INTENTIONALLY MESSY DATA
# ============================================================

titanic = sns.load_dataset("titanic")

# Add 15 intentional duplicate rows
duplicated_rows = titanic.sample(15, random_state=42)
messy_titanic = pd.concat([titanic, duplicated_rows], ignore_index=True)

# Intentionally create inconsistent casing in embark_town
messy_indices = messy_titanic.sample(20, random_state=1).index
messy_titanic.loc[messy_indices, "embark_town"] = (
    messy_titanic.loc[messy_indices, "embark_town"].str.upper()
)

# Second table for the merge
class_info = pd.DataFrame({
    "pclass": [1, 2, 3],
    "class_description": [
        "First Class - Luxury",
        "Second Class - Standard",
        "Third Class - Economy"
    ],
    "deck_access": [
        "Full Access",
        "Limited Access",
        "No Access"
    ]
})

# ============================================================
# STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

rows_before, columns_before = messy_titanic.shape

print(f"\nShape before cleaning: {rows_before} rows x {columns_before} columns")

print("\n--- Missing values ---")
missing_count = messy_titanic.isna().sum()
missing_percent = (messy_titanic.isna().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percent
})

print(missing_report[missing_report["Missing Count"] > 0])

duplicate_count = messy_titanic.duplicated().sum()
print(f"\nNumber of duplicated rows: {duplicate_count}")

print("\n--- Unique values in embark_town ---")
print(messy_titanic["embark_town"].unique())

print("\n--- Initial EDA observations ---")
print(
    f"1. The messy dataset contains {rows_before} rows and {columns_before} columns. "
    "The row count is higher than the original Titanic dataset because duplicate rows were added."
)
print(
    f"2. There are {duplicate_count} exact duplicated rows that should be removed before analysis."
)
print(
    "3. embark_town contains inconsistent text formatting because some values are uppercase "
    "while other values use normal capitalization."
)
print(
    "4. Several columns contain missing values. The main analytical missing values are in "
    "age and embarked, while deck has a much larger amount of missing information."
)

print("\n--- SUMMARY OF PROBLEMS DISCOVERED ---")
print(
    "The dataset contains intentional duplicate records, inconsistent capitalization in "
    "embark_town, missing values in important variables, and a highly incomplete deck column. "
    "These issues can reduce the reliability and consistency of the analysis, so they will "
    "be addressed before building the final visual report."
)

# ============================================================
# STEP 3 - CLEANING AND MERGING
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CLEANING AND MERGING")
print("=" * 70)

# 3.1 Remove duplicates
messy_titanic = messy_titanic.drop_duplicates().copy()
print(f"\nAfter removing duplicates: {messy_titanic.shape}")
print(f"Remaining duplicated rows: {messy_titanic.duplicated().sum()}")

# 3.2 Standardize embark_town
messy_titanic["embark_town"] = messy_titanic["embark_town"].str.title()
print("\nUnique embark_town values after standardization:")
print(messy_titanic["embark_town"].unique())

# 3.3 Fill missing age with the median
# Median is selected because age can contain extreme values and the median is
# less affected by unusually young/old observations than the mean.
age_median = messy_titanic["age"].median()
messy_titanic["age"] = messy_titanic["age"].fillna(age_median)

print(f"\nAge median used for missing values: {age_median:.2f}")
print(f"Remaining missing age values: {messy_titanic['age'].isna().sum()}")

# 3.4 Fill missing embarked with the mode
embarked_mode = messy_titanic["embarked"].mode()[0]
messy_titanic["embarked"] = messy_titanic["embarked"].fillna(embarked_mode)

print(f"\nEmbarked mode used for missing values: {embarked_mode}")
print(f"Remaining missing embarked values: {messy_titanic['embarked'].isna().sum()}")

# 3.5 Drop deck because it has a very high percentage of missing values.
deck_missing_percent = messy_titanic["deck"].isna().mean() * 100
messy_titanic = messy_titanic.drop(columns=["deck"])

print(
    f"\nDeck missing percentage before deletion: {deck_missing_percent:.2f}%"
)
print("Deck column was removed because most of its values are missing.")

# 3.6 Merge with class_info using a LEFT merge
cleaned_titanic = messy_titanic.merge(
    class_info,
    on="pclass",
    how="left"
)

print("\nAfter LEFT merge:")
print(cleaned_titanic[[
    "pclass", "class_description", "deck_access"
]].head())

print(
    f"\nRows after cleaning and merge: {cleaned_titanic.shape[0]}"
)
print(
    f"Columns after cleaning and merge: {cleaned_titanic.shape[1]}"
)

print("\n--- Final validation ---")
print(f"Duplicate rows: {cleaned_titanic.duplicated().sum()}")
print(f"Missing age: {cleaned_titanic['age'].isna().sum()}")
print(f"Missing embarked: {cleaned_titanic['embarked'].isna().sum()}")
print(f"Missing class_description: {cleaned_titanic['class_description'].isna().sum()}")
print(f"Missing deck_access: {cleaned_titanic['deck_access'].isna().sum()}")

# ============================================================
# STEP 4 - VISUAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - VISUAL REPORT")
print("=" * 70)

# ---------------- QUESTION 1 ----------------
# What is the distribution of passengers by class and gender?
plt.figure(figsize=(8, 5))
sns.countplot(
    data=cleaned_titanic,
    x="class_description",
    hue="sex"
)
plt.title("Passenger Distribution by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nQUESTION 1 INTERPRETATION:")
print(
    "The chart shows how passenger counts differ across the three classes and between "
    "male and female passengers. Third class contains the largest passenger group, "
    "and the gender composition also differs across the classes."
)

# ---------------- QUESTION 2 ----------------
# How does ticket price differ by passenger class?
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=cleaned_titanic,
    x="class_description",
    y="fare"
)
plt.title("Ticket Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nQUESTION 2 INTERPRETATION:")
print(
    "Ticket fares are clearly different across passenger classes, with first class "
    "generally showing higher fares than second and third class. The boxplot is useful "
    "because it shows both the typical fare and the spread of fares within each class."
)

# ---------------- QUESTION 3 ----------------
# What is the survival rate by embarkation port?
survival_by_port = (
    cleaned_titanic.groupby("embark_town")["survived"]
    .mean()
    .mul(100)
    .reset_index(name="survival_percentage")
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=survival_by_port,
    x="embark_town",
    y="survival_percentage"
)
plt.title("Survival Percentage by Embarkation Port")
plt.xlabel("Embarkation Port")
plt.ylabel("Survival Percentage (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print("\nQUESTION 3 INTERPRETATION:")
print(
    "The survival percentage varies across the embarkation ports. This suggests that "
    "passengers entering at different ports had different survival outcomes, although "
    "the chart describes an association rather than proving that the port itself caused it."
)

# ---------------- QUESTION 4 ----------------
# Did having family members on board affect survival?
cleaned_titanic["family_onboard"] = (
    (cleaned_titanic["sibsp"] + cleaned_titanic["parch"]) > 0
)

family_survival = (
    cleaned_titanic.groupby("family_onboard")["survived"]
    .mean()
    .mul(100)
    .reset_index(name="survival_percentage")
)

family_survival["family_status"] = family_survival["family_onboard"].map({
    False: "Traveling Alone",
    True: "With Family"
})

plt.figure(figsize=(7, 5))
sns.barplot(
    data=family_survival,
    x="family_status",
    y="survival_percentage"
)
plt.title("Survival Percentage: Traveling Alone vs With Family")
plt.xlabel("Family Status")
plt.ylabel("Survival Percentage (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print("\nQUESTION 4 INTERPRETATION:")
print(
    "The chart compares survival outcomes for passengers traveling alone and passengers "
    "with at least one family member. The two groups have different survival percentages, "
    "showing that family presence was associated with different survival outcomes."
)

# ---------------- QUESTION 5 ----------------
# What is the average age by class and survival status?
cleaned_titanic["survival_status"] = cleaned_titanic["survived"].map({
    0: "Did Not Survive",
    1: "Survived"
})

avg_age = (
    cleaned_titanic.groupby(
        ["class_description", "survival_status"]
    )["age"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=avg_age,
    x="class_description",
    y="age",
    hue="survival_status"
)
plt.title("Average Age by Passenger Class and Survival Status")
plt.xlabel("Passenger Class")
plt.ylabel("Average Age")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nQUESTION 5 INTERPRETATION:")
print(
    "Average age differs across passenger classes and survival groups. The chart allows "
    "the business reader to compare both class and survival status at the same time, "
    "making age patterns easier to identify."
)

# ---------------- REQUIRED INTERACTIVE PLOTLY CHART ----------------
# Interactive version of ticket fare by class
interactive_fare = px.box(
    cleaned_titanic,
    x="class_description",
    y="fare",
    points="all",
    title="Interactive Ticket Fare Distribution by Passenger Class",
    labels={
        "class_description": "Passenger Class",
        "fare": "Ticket Fare"
    }
)

interactive_fare.write_html("titanic_interactive_fare.html")
interactive_fare.show()

print("\nInteractive Plotly chart exported as: titanic_interactive_fare.html")

# ============================================================
# STEP 5 - FINAL DOCUMENTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - FINAL DOCUMENTATION")
print("=" * 70)

rows_after, columns_after = cleaned_titanic.shape

print("\n--- CLEANING DECISIONS ---")
print(
    f"1. Removed {duplicate_count} duplicated rows using drop_duplicates()."
)
print(
    "2. Standardized embark_town using Title Case so values have consistent formatting."
)
print(
    f"3. Filled missing age values with the median ({age_median:.2f}) because "
    "the median is less sensitive to extreme ages."
)
print(
    f"4. Filled missing embarked values with the mode ({embarked_mode}) because "
    "it is a categorical variable."
)
print(
    f"5. Removed deck because {deck_missing_percent:.2f}% of its values were missing, "
    "making the column too incomplete for reliable analysis."
)
print(
    "6. Merged the cleaned Titanic data with class_info using a LEFT merge on pclass."
)

print("\n--- DATASET SIZE COMPARISON ---")
print(
    f"Before cleaning: {rows_before} rows x {columns_before} columns"
)
print(
    f"After cleaning and merge: {rows_after} rows x {columns_after} columns"
)

# Executive summary values
overall_survival = cleaned_titanic["survived"].mean() * 100
family_rates = family_survival.set_index("family_status")["survival_percentage"]

fare_by_class = (
    cleaned_titanic.groupby("class_description")["fare"]
    .mean()
    .sort_values(ascending=False)
)

print("\n--- EXECUTIVE SUMMARY ---")
print(
    f"Overall, approximately {overall_survival:.1f}% of passengers in the cleaned "
    "dataset survived. Survival outcomes also differed between passengers traveling "
    "alone and passengers traveling with family, indicating that travel circumstances "
    "were associated with different outcomes."
)
print(
    "Passenger class was strongly connected with ticket pricing: first-class passengers "
    "paid substantially higher fares on average than passengers in lower classes. "
    "Survival percentages also varied by embarkation port, while age patterns differed "
    "across passenger classes and survival status. These findings provide a concise "
    "business view of the main passenger segments and their outcomes."
)

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)
print("Files created when the script is run:")
print("1. titanic_week2_final_project.py")
print("2. titanic_interactive_fare.html")
print("\nSubmission checklist:")
print("- One Python file containing Steps 1-5")
print("- At least five business-focused visualizations")
print("- At least one interactive Plotly visualization")
print("- HTML file exported from Plotly")
print("- Cleaning decisions and executive summary documented")
