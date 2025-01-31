# Economic Growth Analysis and Prediction for G7 and BRICS Nations

## Overview
This project analyzes historical economic data (1972–2022) and forecasts GDP growth for G7 (Canada, France, Germany, Italy, Japan, UK, USA) and BRICS (Brazil, Russia, India, China, South Africa) nations through 2030. Using exploratory analysis, machine learning, and interactive visualization, the study deciphers growth drivers and provides actionable insights for policymakers and businesses.

## Key Features
- **Dataset**: World Bank data covering 12 countries and 15 parameters, including GDP growth, FDI, inflation, population trends, and unemployment.
- **Preprocessing**: Missing value imputation, outlier removal using IQR, and reshaping for time-series analysis.
- **Methodology**:
  - Exploratory Data Analysis (EDA): Correlation analysis between GDP growth and key drivers.
  - Machine Learning: SVR for G7 (MSE = 0.65), Linear Regression for BRICS (MSE = 0.72).
  - Time-Series Forecasting: Predictions from 2023–2030.
- **Interactive Dashboard**: Built with Streamlit and Plotly for visualizing trends and comparing metrics across countries.

## Key Insights
- G7 Challenges: Aging populations and market saturation correlate with slower growth.
- BRICS Opportunities: FDI inflows and trade openness drive higher growth potential.
- Global Shift: BRICS' rising influence signals a rebalancing of economic power.

## Tools Used
- Python Libraries: Pandas, Scikit-learn, Matplotlib, Streamlit, Plotly.

## Resources

- **Report**: 
  [Link](http://127.0.0.1:5500/uploads/Final%20Report.pdf)
  
- **Dashboard**: 
  [Link](https://g7-vs-brics-q6rli72jrpgndzfyfsfnwc.streamlit.app/)
  
- **Dataset**: 
  [Link](https://drive.google.com/file/d/1w4-KpYewybTSRncCyEKByz9L_ebEzitc/view?usp=sharing)

- **Data Source**: 
  [Link](https://example.com/data-source)
  
- **Tutorial Explanation**: 
  [Link](http://127.0.0.1:5500/uploads/Videos/G7%20VS%20BRICS.mp4)

## Future Work
- Incorporate climate data, geopolitical indices, or sector-specific metrics.
- Expand country coverage to include emerging economies like Indonesia or Mexico.

## Conclusion
This project demonstrates the transformative potential of data science in economics, providing a replicable framework for forecasting and policy analysis.


