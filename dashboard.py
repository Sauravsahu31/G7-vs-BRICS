import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS to reduce spacing
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stSlider, .stSelectbox {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('New GB.csv')  # Replace with your actual data file
    for col in range(1972, 2023):
        df[str(col)] = pd.to_numeric(df[str(col)], errors='coerce')
    year_cols = [str(i) for i in range(1972, 2023)]
    df[year_cols] = df[year_cols].apply(lambda x: x.fillna(x.mean()), axis=1)
    df_melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Parameters'], var_name='Year', value_name='Value')
    return df_melted

df_melted = load_data()

# Define country groups
g7_countries = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
brics_countries = ['Brazil', 'Russian Federation', 'India', 'China', 'South Africa']

# Title
st.title('Economic Data Analysis Dashboard')

# Create a container for all sections
main_container = st.container()

with main_container:
    # Section 1: Predicted GDP Growth
    st.header('Predicted GDP Growth (2023-2030)')
    
    # Example prediction data
    g7_pred = [4.79, 4.77, 4.74, 4.71, 4.68, 4.66, 4.65, 4.64]
    brics_pred = [6.57, 6.60, 6.63, 6.66, 6.69, 6.72, 6.75, 6.78]
    years = np.arange(2023, 2031)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=years, y=g7_pred, mode='lines+markers', name='G7 Predicted GDP Growth', 
                                 line=dict(color='royalblue', width=2), marker=dict(size=8, color='lightblue')))
    fig_pred.add_trace(go.Scatter(x=years, y=brics_pred, mode='lines+markers', name='BRICS Predicted GDP Growth', 
                                 line=dict(color='tomato', width=2), marker=dict(size=8, color='salmon')))
    fig_pred.update_layout(
        title='Predicted Average GDP Growth (2023-2030)',
        xaxis_title='Year',
        yaxis_title='Predicted GDP Growth (annual %)',
        template='plotly_white',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Section 2: Average Comparison
    st.header('Average Comparative Analysis')
    col1, col2 = st.columns(2)
    with col1:
        parameter = st.selectbox('Select Parameter:', df_melted['Parameters'].unique(), index=0)
    with col2:
        year_range = st.slider('Select Year Range:', 1972, 2022, (1972, 2022))
    
    filtered_df = df_melted[(df_melted['Parameters'] == parameter) & 
                           (df_melted['Year'].astype(int).between(year_range[0], year_range[1]))]
    
    g7_avg = filtered_df[filtered_df['Country Name'].isin(g7_countries)].groupby('Year')['Value'].mean().reset_index()
    brics_avg = filtered_df[filtered_df['Country Name'].isin(brics_countries)].groupby('Year')['Value'].mean().reset_index()

    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=g7_avg['Year'], y=g7_avg['Value'], mode='lines+markers', name='G7 Average', 
                                line=dict(color='mediumseagreen', width=2), marker=dict(size=8, color='lightgreen')))
    fig_avg.add_trace(go.Scatter(x=brics_avg['Year'], y=brics_avg['Value'], mode='lines+markers', name='BRICS Average', 
                                line=dict(color='orange', width=2), marker=dict(size=8, color='gold')))
    fig_avg.update_layout(
        title=f'Average {parameter} from {year_range[0]} to {year_range[1]}',
        xaxis_title='Years',
        yaxis_title='Average Values',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # Section 3: Country Comparison
    st.header('Country Comparison Analysis')
    c1, c2 = st.columns(2)
    with c1:
        country1 = st.selectbox('Select First Country:', df_melted['Country Name'].unique(), index=0)
    with c2:
        country2 = st.selectbox('Select Second Country:', df_melted['Country Name'].unique(), index=1)
    
    col3, col4 = st.columns(2)
    with col3:
        parameter_comp = st.selectbox('Select Parameter:', df_melted['Parameters'].unique(), index=0, key='param_comp')
    with col4:
        year_range_comp = st.slider('Select Year Range:', 1972, 2022, (1972, 2022), key='year_comp')

    filtered_df_comp = df_melted[(df_melted['Parameters'] == parameter_comp) & 
                                (df_melted['Year'].astype(int).between(year_range_comp[0], year_range_comp[1]))]
    
    country1_data = filtered_df_comp[filtered_df_comp['Country Name'] == country1]
    country2_data = filtered_df_comp[filtered_df_comp['Country Name'] == country2]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=country1_data['Year'], y=country1_data['Value'], mode='lines+markers', name=country1, 
                                 line=dict(color='purple', width=2), marker=dict(size=8, color='violet')))
    fig_comp.add_trace(go.Scatter(x=country2_data['Year'], y=country2_data['Value'], mode='lines+markers', name=country2, 
                                 line=dict(color='darkorange', width=2), marker=dict(size=8, color='peachpuff')))
    fig_comp.update_layout(
        title=f'Comparison of {country1} and {country2}',
        xaxis_title='Years',
        yaxis_title='Values',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_comp, use_container_width=True)