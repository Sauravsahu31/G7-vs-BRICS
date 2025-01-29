# G7 VS BRICS :2030 
Economic Data Analysis 

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

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/economic-data-analysis.git
   cd economic-data-analysis
2. **Install the required dependencies:**

pip install -r requirements.txt
Run the dashboard: python app.py
Access the dashboard: Open your web browser and go to http://127.0.0.1:8050/ to view the dashboard.

**Usage**
Predicted GDP Growth (2023-2030): The dashboard displays the predicted GDP growth for both G7 and BRICS countries from 2023 to 2030.

Average Comparative Analysis: Compare the average values of different economic parameters between G7 and BRICS countries over a selected time range.

Country Comparison: Compare specific economic parameters between two selected countries over a selected time range.

**Results**

Best Model for G7: SVR

Best Model for BRICS: Linear Regression

Predicted GDP Growth (2023-2030):

G7: Predicted to grow at an average rate of around 4.7% to 4.8%.

BRICS: Predicted to grow at an average rate of around 6.5% to 6.8%.

**Contributing**

Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.

**License**

This project is licensed under the MIT License. See the LICENSE file for more details.

**Acknowledgments**

Data sourced from the World Bank.

Special thanks to the developers of Dash, Plotly, and Scikit-learn for their amazing libraries.


