import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# Load the dataset
df = pd.read_csv('New GB.csv')  # Replace with your actual data file

# Convert columns from 1972 to 2022 to numeric
for col in range(1972, 2023):
    df[str(col)] = pd.to_numeric(df[str(col)], errors='coerce')

# Melt the dataset
df_melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Parameters'], var_name='Year', value_name='Value')

# Define country groups
g7_countries = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
brics_countries = ['Brazil', 'Russian Federation', 'India', 'China', 'South Africa']

# Get unique country names for dropdown options
unique_countries = df['Country Name'].unique()

# Create a Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    html.H1('Economic Data Analysis Dashboard', className='text-center', style={'color': '#007BFF'}),
    
    # Section for displaying prediction results
    html.Div([
        html.H2('Predicted GDP Growth (2023-2030)', className='text-center', style={'color': '#FF5733'}),
        
        dcc.Graph(id='prediction-graph', style={'height': '60vh'}),
        
        html.Div(id='prediction-table', style={'overflowX': 'auto'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Section for average comparison of G7 and BRICS
    html.Div([
        html.H2('Average Comparative Analysis of G7 and BRICS', className='text-center', style={'color': '#28a745'}),
        
        html.Label('Select Parameter:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='parameter-dropdown',
            options=[{'label': param, 'value': param} for param in df['Parameters'].unique()],
            value='GDP growth (annual %)',
            style={'color': '#007BFF'}
        ),
        
        html.Label('Select Year Range:', style={'fontWeight': 'bold'}),
        dcc.RangeSlider(
            id='year-slider',
            min=1972,
            max=2022,
            value=[1972, 2022],  # Ensure this has a valid default value
            marks={i: str(i) for i in range(1972, 2023, 2)},
            step=1
        ),
        
        dcc.Graph(id='average-comparison-graph', style={'height': '60vh'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Section for comparing two specific countries
    html.Div([
        html.H2('Country Comparison Analysis', className='text-center', style={'color': '#17a2b8'}),
        
        dbc.Row([
            dbc.Col([
                html.Label('Select First Country:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='country1-dropdown',
                    options=[{'label': country, 'value': country} for country in unique_countries],
                    value=unique_countries[0],
                    style={'color': '#007BFF'}
                )
            ], width=6),
            
            dbc.Col([
                html.Label('Select Second Country:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='country2-dropdown',
                    options=[{'label': country, 'value': country} for country in unique_countries],
                    value=unique_countries[1],
                    style={'color': '#007BFF'}
                )
            ], width=6)
        ]),
        
        html.Label('Select Parameter:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='country-parameter-dropdown',
            options=[{'label': param, 'value': param} for param in df['Parameters'].unique()],  # Corrected here
            value='GDP growth (annual %)',
            style={'color': '#007BFF'}
        ),
        
        html.Label('Select Year Range:', style={'fontWeight': 'bold'}),
        dcc.RangeSlider(
            id='country-year-slider',
            min=1972,
            max=2022,
            value=[1972, 2022],
            marks={i: str(i) for i in range(1972, 2023, 2)},
            step=1
        ),
        
        dcc.Graph(id='comparison-graph', style={'height': '60vh'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'})
], fluid=True)

# Define callback for prediction graph
@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('parameter-dropdown', 'value')]
)
def update_prediction_graph(selected_parameter):
    # Placeholder for prediction logic
    years = np.arange(2023, 2031)
    g7_pred = np.random.uniform(4, 6, size=len(years))  # Dummy data for G7 predictions
    brics_pred = np.random.uniform(5, 7, size=len(years))  # Dummy data for BRICS predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=g7_pred, mode='lines+markers', name='G7 Predicted GDP Growth', line=dict(color='blue', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=years, y=brics_pred, mode='lines+markers', name='BRICS Predicted GDP Growth', line=dict(color='orange', width=2), marker=dict(size=8)))

    fig.update_layout(
        title='Predicted Average GDP Growth (2023-2030)',
        xaxis_title='Year',
        yaxis_title='Predicted GDP Growth (annual %)',
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    return fig

# Define callback for average comparison graph
@app.callback(
    Output('average-comparison-graph', 'figure'),
    [Input('parameter-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_average_graph(selected_parameter, selected_years):
    filtered_df = df_melted[(df_melted['Parameters'] == selected_parameter) &
                             (df_melted['Year'].astype(int).between(selected_years[0], selected_years[1]))]

    g7_avg = filtered_df[filtered_df['Country Name'].isin(g7_countries)].groupby('Year')['Value'].mean().reset_index()
    brics_avg = filtered_df[filtered_df['Country Name'].isin(brics_countries)].groupby('Year')['Value'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g7_avg['Year'], y=g7_avg['Value'], mode='lines+markers', name='G7 Average', line=dict(color='blue', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=brics_avg['Year'], y=brics_avg['Value'], mode='lines+markers', name='BRICS Average', line=dict(color='orange', width=2), marker=dict(size=8)))

    fig.update_layout(
        title=f'Average {selected_parameter} from {selected_years[0]} to {selected_years[1]}',
        xaxis_title='Years',
        yaxis_title='Average Values',
        xaxis=dict(tickmode='linear', dtick=2, tickangle=-45),
        yaxis=dict(range=[0, max(g7_avg['Value'].max(), brics_avg['Value'].max()) * 1.1]),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    return fig

# Define callback for country comparison graph
@app.callback(
    Output('comparison-graph', 'figure'),
    [Input('country1-dropdown', 'value'),
     Input('country2-dropdown', 'value'),
     Input('country-parameter-dropdown', 'value'),
     Input('country-year-slider', 'value')]
)
def update_comparison_graph(country1, country2, selected_parameter, selected_years):
    filtered_df = df_melted[(df_melted['Parameters'] == selected_parameter) &
                             (df_melted['Year'].astype(int).between(selected_years[0], selected_years[1]))]

    country1_data = filtered_df[filtered_df['Country Name'] == country1]
    country2_data = filtered_df[filtered_df['Country Name'] == country2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=country1_data['Year'], y=country1_data['Value'], mode='lines+markers', name=country1, line=dict(color='green', width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=country2_data['Year'], y=country2_data['Value'], mode='lines+markers', name=country2, line=dict(color='red', width=2), marker=dict(size=8)))

    fig.update_layout(
        title=f'Comparison of {country1} and {country2} for {selected_parameter}',
        xaxis_title='Years',
        yaxis_title='Values',
        xaxis=dict(tickmode='linear', dtick=2, tickangle=-45),
        yaxis=dict(range=[0, max(country1_data['Value'].max(), country2_data['Value'].max()) * 1.1]),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)