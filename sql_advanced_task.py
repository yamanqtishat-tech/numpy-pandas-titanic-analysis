import sqlite3

# Connect to the database
conn = sqlite3.connect("store.db")
cursor = conn.cursor()

print("=" * 60)
print("PART 1 - JOINs")
print("=" * 60)

# 1. INNER JOIN
print("\n1. INNER JOIN - Sales with Product Information")

query1 = """
SELECT
    p.product_name,
    s.quantity,
    s.sale_date
FROM sales s
INNER JOIN products p
    ON s.product_id = p.product_id
ORDER BY s.sale_date;
"""

cursor.execute(query1)

for row in cursor.fetchall():
    print(row)


# 2. Total Revenue per Product
print("\n2. Total Revenue per Product")

query2 = """
SELECT
    p.product_name,
    SUM(s.quantity * p.price) AS total_revenue
FROM sales s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_revenue DESC;
"""

cursor.execute(query2)

for row in cursor.fetchall():
    print(row)


# 3. LEFT JOIN - Show all products, including unsold products
print("\n3. LEFT JOIN - All Products and Sales")

query3 = """
SELECT
    p.product_name,
    s.quantity,
    s.sale_date
FROM products p
LEFT JOIN sales s
    ON p.product_id = s.product_id
ORDER BY p.product_id;
"""

cursor.execute(query3)

for row in cursor.fetchall():
    print(row)


# Find unsold products
print("\nUnsold Products:")

query_unsold = """
SELECT p.product_name
FROM products p
LEFT JOIN sales s
    ON p.product_id = s.product_id
WHERE s.product_id IS NULL;
"""

cursor.execute(query_unsold)

unsold_products = cursor.fetchall()

if unsold_products:
    for row in unsold_products:
        print(row[0])
else:
    print("No unsold products.")


# ============================================================
# PART 2 - SUBQUERIES
# ============================================================

print("\n" + "=" * 60)
print("PART 2 - SUBQUERIES")
print("=" * 60)


# 1. Products priced above average
print("\n1. Products Priced Above Average")

query4 = """
SELECT
    product_name,
    price
FROM products
WHERE price > (
    SELECT AVG(price)
    FROM products
)
ORDER BY price DESC;
"""

cursor.execute(query4)

for row in cursor.fetchall():
    print(row)


# ============================================================
# PART 3 - JOIN EXPLANATION
# ============================================================

print("\n4. INNER JOIN vs LEFT JOIN")

print("INNER JOIN returns only records that have matching values in both tables.")

print("LEFT JOIN returns all records from the left table, even when there is no matching record in the right table.")

# ============================================================
# PART 2 - MORE SUBQUERIES
# ============================================================

# 2. Sales with quantity above average
print("\n2. Sales with Quantity Above Average")

query5 = """
SELECT
    product_id,
    quantity,
    sale_date
FROM sales
WHERE quantity > (
    SELECT AVG(quantity)
    FROM sales
)
ORDER BY quantity DESC;
"""

cursor.execute(query5)

for row in cursor.fetchall():
    print(row)


# 3. Most Expensive Product
print("\n3. Most Expensive Product")

query6 = """
SELECT
    product_name,
    price
FROM products
WHERE price = (
    SELECT MAX(price)
    FROM products
);

"""

cursor.execute(query6)

for row in cursor.fetchall():
    print(row)


    # ============================================================
# PART 3 - WINDOW FUNCTIONS
# ============================================================

print("\n" + "=" * 60)
print("PART 3 - WINDOW FUNCTIONS")
print("=" * 60)

# 1. Rank products by total revenue
print("\n1. Product Ranking by Total Revenue")

query7 = """
SELECT
    product_name,
    total_revenue,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM (
    SELECT
        p.product_name,
        SUM(s.quantity * p.price) AS total_revenue
    FROM sales s
    JOIN products p
        ON s.product_id = p.product_id
    GROUP BY p.product_name
)
ORDER BY revenue_rank;
"""

cursor.execute(query7)

for row in cursor.fetchall():
    print(row)


# 2. Running Total of Sales
print("\n2. Running Total of Sales")

query8 = """
SELECT
    sale_date,
    quantity,
    SUM(quantity) OVER (ORDER BY sale_date) AS running_total
FROM sales
ORDER BY sale_date;
"""

cursor.execute(query8)

for row in cursor.fetchall():
    print(row)


    # 3. Largest Sale in Each City
print("\n3. Largest Sale in Each City")

query9 = """
SELECT
    city,
    product_id,
    quantity,
    sale_date,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY quantity DESC
    ) AS city_rank
FROM sales
ORDER BY city, city_rank;
"""

cursor.execute(query9)

for row in cursor.fetchall():
    print(row)


# ============================================================
# PART 4 - SQL WITH PANDAS
# ============================================================

print("\n" + "=" * 60)
print("PART 4 - SQL WITH PANDAS")
print("=" * 60)

import pandas as pd

# Get total revenue as a DataFrame
revenue_query = """
SELECT
    p.product_name,
    SUM(s.quantity * p.price) AS total_revenue
FROM sales s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_revenue DESC;
"""

revenue_df = pd.read_sql_query(revenue_query, conn)

print("\nRevenue DataFrame:")
print(revenue_df)

# Show top 5 products
print("\nTop 5 Products by Revenue:")
print(revenue_df.head(5))


# ============================================================
# Bar Chart - Top 5 Products by Revenue
# ============================================================

import matplotlib.pyplot as plt

top5 = revenue_df.head(5)

plt.figure(figsize=(10, 6))
plt.bar(top5["product_name"], top5["total_revenue"])
plt.xlabel("Product")
plt.ylabel("Total Revenue")
plt.title("Top 5 Products by Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# Running Total with Pandas
# ============================================================

running_total_query = """
SELECT
    sale_date,
    quantity,
    SUM(quantity) OVER (ORDER BY sale_date) AS running_total
FROM sales
ORDER BY sale_date;
"""

running_total_df = pd.read_sql_query(running_total_query, conn)

print("\nRunning Total DataFrame:")
print(running_total_df)

# Line Chart - Running Total
plt.figure(figsize=(10, 6))
plt.plot(
    running_total_df["sale_date"],
    running_total_df["running_total"],
    marker="o"
)
plt.xlabel("Sale Date")
plt.ylabel("Running Total")
plt.title("Running Total of Sales Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nSales Table Columns:")
cursor.execute("PRAGMA table_info(sales)")
for row in cursor.fetchall():
    print(row)

print("\nProducts Table Columns:")
cursor.execute("PRAGMA table_info(products)")
for row in cursor.fetchall():
    print(row)

# ============================================================
# PART 5 - INTEGRATED SQL + PANDAS
# ============================================================

print("\n" + "=" * 60)
print("PART 5 - INTEGRATED ANALYSIS")
print("=" * 60)

integrated_query = """
SELECT
    s.city,
    p.product_name,
    SUM(s.quantity * p.price) AS total_revenue,
    RANK() OVER (
        PARTITION BY s.city
        ORDER BY SUM(s.quantity * p.price) DESC
    ) AS revenue_rank
FROM sales s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY s.city, p.product_name
ORDER BY s.city, revenue_rank;
"""

integrated_df = pd.read_sql_query(integrated_query, conn)

print("\nRevenue Ranking by City:")
print(integrated_df)

# Plot
plt.figure(figsize=(10, 6))

for city in integrated_df["city"].unique():
    city_data = integrated_df[integrated_df["city"] == city]
    plt.plot(
        city_data["product_name"],
        city_data["total_revenue"],
        marker="o",
        label=city
    )

plt.xlabel("Product")
plt.ylabel("Total Revenue")
plt.title("Product Revenue by City")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("\nInsight:")
print("The analysis shows the revenue ranking of products within each city.")

# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()

print("\nDatabase connection closed.")





