# G7 VS BRICES :2030 Economic Data Analysis Dashboard

## Overview

This project analyzes economic data from 12 countries (G7 and BRICS) over the years 1972 to 2022. The dataset includes 15 different economic parameters such as GDP growth, fertility rate, inflation, and more. The goal of the project is to predict the average GDP growth (annual %) for the G7 and BRICS groups from 2023 to 2030 using machine learning models.

## Dataset

The dataset was downloaded from the World Bank website and includes the following parameters:

1. Current account balance (% of GDP)
2. Fertility rate, total (births per woman)
3. Foreign direct investment, net inflows (% of GDP)
4. Foreign direct investment, net outflows (% of GDP)
5. GDP (current US$)
6. GDP growth (annual %)
7. GDP per capita (current US$)
8. GDP per capita growth (annual %)
9. Gross savings (% of GDP)
10. Imports of goods and services (% of GDP)
11. Inflation, consumer prices (annual %)
12. Population ages 65 and above (% of total population)
13. Population growth (annual %)
14. Population, total
15. Unemployment, total (% of total labor force) (modeled ILO estimate)

## Project Structure

- **Data Preparation**: The dataset was cleaned, missing values were handled, and outliers were removed. The data was then melted into a long format for easier analysis.
- **Machine Learning Models**: Five different machine learning models were used to predict GDP growth:
  - Random Forest
  - Linear Regression
  - Decision Tree
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning**: GridSearchCV was used for hyperparameter tuning to find the best model for each group (G7 and BRICS).
- **Prediction**: The best models were used to predict the average GDP growth for the years 2023 to 2030.
- **Dashboard**: An interactive dashboard was created using Dash and Plotly to visualize the results.
