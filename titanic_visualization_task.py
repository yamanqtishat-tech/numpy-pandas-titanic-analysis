# Titanic Data Visualization Task - Parts 1 to 4
# Uses the same Titanic dataset from previous tasks.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# ============================================================
# PART 1 - MATPLOTLIB
# ============================================================

# 1. Histogram of age
plt.figure(figsize=(8, 5))
plt.hist(titanic["age"].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

print("PART 1 - Age Histogram Interpretation:")
print("The age distribution is concentrated mainly around young and middle-aged passengers,")
print("with fewer passengers at very young and very old ages. The distribution is spread")
print("across a relatively wide age range rather than being centered at one single age.")

# 2. Bar chart - number of passengers by passenger class
pclass_counts = titanic["pclass"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(pclass_counts.index.astype(str), pclass_counts.values)
plt.title("Number of Passengers by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

print("\nPART 1 - Bar Chart Interpretation:")
print("The third class contains the largest number of passengers, followed by second class,")
print("while first class contains the fewest passengers.")

# 3. Scatter plot - age vs fare
plt.figure(figsize=(8, 5))
plt.scatter(titanic["age"], titanic["fare"], alpha=0.5)
plt.title("Relationship Between Age and Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

print("\nPART 1 - Scatter Plot Interpretation:")
print("There is no strong direct visual relationship between age and fare. Fare values vary")
print("considerably across passengers of similar ages, although some very high fares are visible.")

# ============================================================
# PART 2 - SEABORN
# ============================================================

# 1. Correlation matrix and heatmap
numeric_data = titanic.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

print("\nPART 2 - Correlation Matrix:")
print(correlation_matrix.round(3))

plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Titanic Variables")
plt.show()

# Find strongest correlation excluding the diagonal
corr_pairs = correlation_matrix.where(
    ~pd.DataFrame(
        [[i == j for j in correlation_matrix.columns] for i in correlation_matrix.columns],
        index=correlation_matrix.index,
        columns=correlation_matrix.columns
    )
)
strongest_pair = corr_pairs.abs().stack().sort_values(ascending=False).index[0]
strongest_value = correlation_matrix.loc[strongest_pair[0], strongest_pair[1]]

print("\nStrongest correlation (excluding variables with themselves):")
print(f"{strongest_pair[0]} and {strongest_pair[1]} = {strongest_value:.3f}")
print("Interpretation: Passenger class and fare show a relatively strong negative relationship;")
print("lower class numbers represent higher classes and are associated with higher fares.")

# 2. Boxplot of age by passenger class
plt.figure(figsize=(8, 5))
sns.boxplot(data=titanic, x="pclass", y="age")
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.show()

mean_age_by_class = titanic.groupby("pclass")["age"].mean()
print("\nPART 2 - Mean age by passenger class:")
print(mean_age_by_class.round(2))

print("Interpretation: First-class passengers are older on average than second-class")
print("passengers, while third-class passengers have the youngest average age.")

# 3. Countplot - survival by sex
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x="sex", hue="survived")
plt.title("Survival Status by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

print("\nPART 2 - Countplot Interpretation:")
print("The plot shows a much larger number of female survivors relative to female non-survivors,")
print("while male non-survivors greatly outnumber male survivors.")

# 4. Pairplot
pair_data = titanic[["age", "fare", "pclass"]].dropna()

sns.pairplot(pair_data)
plt.show()

print("\nPART 2 - Pairplot Interpretation:")
print("The most visible pattern is the relationship between passenger class and fare.")
print("Higher-class passengers generally paid higher fares. Age does not show a strong")
print("linear relationship with fare.")

# ============================================================
# PART 3 - PLOTLY
# ============================================================

# 1. Interactive bar chart
pclass_df = pclass_counts.reset_index()
pclass_df.columns = ["pclass", "passenger_count"]

fig1 = px.bar(
    pclass_df,
    x="pclass",
    y="passenger_count",
    title="Interactive Number of Passengers by Passenger Class",
    labels={
        "pclass": "Passenger Class",
        "passenger_count": "Number of Passengers"
    }
)
fig1.show()

# 2. Interactive scatter plot - age vs fare colored by survived
scatter_data = titanic[["age", "fare", "survived"]].dropna()

fig2 = px.scatter(
    scatter_data,
    x="age",
    y="fare",
    color="survived",
    title="Interactive Relationship Between Age and Fare by Survival",
    labels={
        "age": "Age",
        "fare": "Fare",
        "survived": "Survived"
    },
    hover_data=["age", "fare", "survived"]
)
fig2.show()

# 3. Interactive histogram of age by sex
age_data = titanic[["age", "sex"]].dropna()

fig3 = px.histogram(
    age_data,
    x="age",
    color="sex",
    nbins=20,
    title="Interactive Age Distribution by Gender",
    labels={
        "age": "Age",
        "sex": "Gender"
    }
)
fig3.show()

# 4. Export one Plotly chart as HTML
fig2.write_html("chart.html")
print("\nPART 3: chart.html has been created in the same folder as this Python file.")

# ============================================================
# PART 4 - CHOOSING THE RIGHT CHART FOR BUSINESS QUESTIONS
# ============================================================

# Question 1:
# "Does ticket price differ considerably between the three classes?"
#
# Best chart: Boxplot
# Reason: We are comparing distributions of a numerical variable (fare)
# across three categorical groups (pclass).

plt.figure(figsize=(8, 5))
sns.boxplot(data=titanic, x="pclass", y="fare")
plt.title("Ticket Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

print("\nPART 4 - Question 1:")
print("The boxplot is appropriate because it compares the distribution of a numerical variable")
print("across three categories. Ticket fares differ substantially between classes, with")
print("first class generally having much higher fares than second and third class.")
print("A line chart would not be appropriate because passenger class is categorical, not time-based.")

# Question 2:
# "What is the overall percentage of survivors and non-survivors?"
#
# Best chart: Pie chart / bar chart of percentages
# Here we use a bar chart because exact percentages are easy to compare.

survival_pct = titanic["survived"].value_counts(normalize=True).sort_index() * 100
survival_pct.index = ["Did Not Survive", "Survived"]

plt.figure(figsize=(8, 5))
plt.bar(survival_pct.index, survival_pct.values)
plt.title("Overall Survival Percentage")
plt.xlabel("Survival Status")
plt.ylabel("Percentage (%)")

for i, value in enumerate(survival_pct.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.show()

print("\nPART 4 - Question 2:")
print(f"Approximately {survival_pct['Survived']:.1f}% of passengers survived,")
print(f"while approximately {survival_pct['Did Not Survive']:.1f}% did not survive.")
print("A percentage bar chart is appropriate because the question compares parts of the total.")
print("A scatter plot would not be suitable because there are no two continuous variables to relate.")

# Question 3:
# "Is there a relationship between family size on board and probability of survival?"
#
# New variable: family_size = sibsp + parch
# Best chart: bar chart of mean survival rate by family size.

titanic["family_size"] = titanic["sibsp"] + titanic["parch"]

family_survival = titanic.groupby("family_size", as_index=False)["survived"].mean()
family_survival["survival_percentage"] = family_survival["survived"] * 100

plt.figure(figsize=(10, 5))
plt.bar(
    family_survival["family_size"],
    family_survival["survival_percentage"]
)
plt.title("Survival Percentage by Family Size")
plt.xlabel("Family Size (SibSp + Parch)")
plt.ylabel("Survival Percentage (%)")
plt.show()

print("\nPART 4 - Question 3:")
print("A bar chart is appropriate because family size is grouped into discrete values and")
print("we are comparing the average survival percentage across those groups.")
print("The relationship is not perfectly linear: survival percentages vary across family sizes,")
print("and some larger family-size groups contain relatively few passengers.")
print("A scatter plot is less suitable here because the business question focuses on group-level survival rates.")

# Question 4:
# "Compare the average ticket fare between different ports of embarkation."
#
# Best chart: Bar chart
# Reason: embarked is categorical and fare is numerical.

embarked_fare = titanic.groupby("embarked", as_index=False)["fare"].mean()

plt.figure(figsize=(8, 5))
plt.bar(embarked_fare["embarked"], embarked_fare["fare"])
plt.title("Average Ticket Fare by Port of Embarkation")
plt.xlabel("Port of Embarkation")
plt.ylabel("Average Fare")
plt.show()

print("\nPART 4 - Question 4:")
print("The average fare differs between the embarkation ports.")
print("A bar chart is appropriate because it compares one numerical measure (average fare)")
print("across categorical groups (embarked). A histogram would show the distribution of fares")
print("but would not directly compare the group averages.")

print("\nAverage fare by embarkation port:")
print(embarked_fare.round(2))

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL SUMMARY")
print("============================================================")
print("1. Age is spread across a wide range, with more passengers concentrated in young/middle ages.")
print("2. Third class contains the largest number of passengers.")
print("3. Age and fare do not show a strong direct visual relationship.")
print("4. Passenger class and fare have a relatively strong negative correlation.")
print("5. First-class passengers are older on average than third-class passengers.")
print("6. Survival differs substantially by gender.")
print("7. Ticket fares differ considerably between passenger classes.")
print("8. Overall survival was about 38%, while about 62% did not survive.")
print("9. Family size and survival show variation, but the pattern is not simply linear.")
print("10. Average fares differ between embarkation ports.")
print("11. The interactive Plotly scatter plot was exported to chart.html.")
