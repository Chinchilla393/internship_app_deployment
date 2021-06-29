# Import all necessary libraries

# from os import startfile
import pandas as pd
from glob import glob
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from datetime import date
from dash.dependencies import *
#---------------------------------------------------------------------------------------------

# Read the graph
csv_files = sorted(glob('csv files/Chart*.csv'))
energy_data = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index = True)
energy_data.to_csv('Energy Data.csv', index = False)

energy = pd.read_csv('Energy Data.csv')
energy.rename(columns = {'System Production (W)' : 'prod', 'Time' : 'Date'}, inplace = True)
energy['Date'] = pd.to_datetime(energy['Date'], format = '%m/%d/%Y %H:%M')
energy.sort_values(by = 'Date', inplace = True, ignore_index = True)
energy.set_index('Date',inplace = True)
#---------------------------------------------------------------------------------------------
# Resampling graphs by: hour; day; month; year
energy_hour = energy.resample('H').sum()
energy_day = energy.resample('D').sum()
energy_month = energy.resample('M').sum()
energy_year = energy.resample('Y').sum()

#---------------------------------------------------------------------------------------------
# Reading consumption graphs:
consumption_weekday = pd.read_csv('consumption_weekdays.csv')
consumption_weekend = pd.read_csv('consumption_weekends.csv')
#---------------------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#---------------------------------------------------------------------------------------------


# Layout
app.layout = html.Div(children=[
    html.H1(children='RTU Kipsala Solar Energy Production',
            style={'textAlign':'center'}
            ),
        html.Br(),

    html.Div([

        # Left side of app
        html.Div([

            # Pick sample frequency (date, month, year)
            dcc.RadioItems(id='radio-items-resample-frequency',
                       options=[
                           {'label':'by Day', 'value':'day_freq'},
                           {'label':'by Month', 'value':'month_freq'},
                           {'label':'by Year', 'value':'year_freq'},
                       ],
                       value='day_freq',
                       labelStyle={'display':'inline-block'}
            ),

            # Pick start - end dates with btn
            dcc.DatePickerRange(
                id='pick-the-damn-dates',
                min_date_allowed=date(2018, 11, 26),
                max_date_allowed=date(2021, 5, 30),
                initial_visible_month=date(2021, 3, 1),
                # start_date=date(2018, 11, 26),
                # end_date=date(2021, 5, 30),
                end_date_placeholder_text='End Date',
                start_date_placeholder_text='Start Date',
                display_format='DD MMM, YYYY',
                with_portal=True,
                number_of_months_shown=3,
                minimum_nights=7,
                clearable=True,

            ),
            html.Br(),
            html.Button('Apply Dates', id='apply-date-change-button', n_clicks=0),

            # Download data btn
            html.Button('Download CSV', id='btn_one_csv'),
            dcc.Download(id='download-dataframe-csv-one'),

            # Graph box
            dcc.Graph(
            id='main-figure-update',
            figure={},
            config={
                'staticPlot':False,
                'scrollZoom':True,
                'responsive':True,
                    },
            
            
        ),
        html.Div([
            html.H3('SOLAR PRODUCTION INFO DATA'),
            html.H5(id='average-prod'),
            html.H5(id='min-prod'),
            html.H5(id='max-prod'),
            html.H5(id='sum-prod'),
            html.H4(id='comparison-prod'),
        ]),
        ], className='seven columns'),
        

        # Right side of app
        html.Div([
            html.Br(),
            # Input consumption btn
            dcc.Input(
                id='input-consumption', 
                type='number', 
                placeholder='Input Consumption'
            ),
            
            html.Button('Apply Consumption', id='apply-consumption-change-button', n_clicks=0),
           
            html.Br(),

            # Download csv btn
            html.Button('Download CSV', id='btn_two_csv'),
            dcc.Download(id = 'download-dataframe-csv-two'),
            

            # Upper graph box
            dcc.Graph(
            id='secondary-graph-one-update',
            figure={},
        
            ),

            # Lower graph box
            dcc.Graph(
            id='secondary-graph-two-update',
            figure={},
        
            ),
        ], className='five columns')
        
        
    ])
])

#---------------------------------------------------------------------------------------------
### Callback to update main figure

@app.callback(
     Output('main-figure-update', 'figure'),
    [Input('apply-date-change-button', 'n_clicks'),
     Input('radio-items-resample-frequency', 'value'),
     State('pick-the-damn-dates', 'start_date'),
     State('pick-the-damn-dates', 'end_date'),
    ]
)
def update_output(apply_btn,freq_val, start_date, end_date):


    global df_one
# Day freq graph    
    if freq_val=='day_freq':
        if (start_date==None and end_date==None):
            df_one = energy_day       
        else:
            df_one = energy_day.loc[start_date:end_date]
        
    
        energy_day_fig = px.line(df_one, 
                         x=df_one.index, 
                         y=df_one['prod'],
                         labels={'y':'Solar Energy Production, W'},
                         height=700,
                        )

        energy_day_fig.update_layout(
            hovermode='x',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,label="1d",step="day",stepmode="backward"),   
                        dict(count=7,label="1w",step="day",stepmode="backward"),    
                        dict(count=1,label='1m',step='month',stepmode='backward'),      
                        dict(count=4,label="1q",step="month",stepmode="backward"),
                        dict(count=1,label="1y",step="year",stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        return energy_day_fig

# Month freq graph  
    elif freq_val=='month_freq':
        df_one = energy_month
        energy_month_fig = px.bar(x=df_one.index, y=df_one['prod'], 
                          color=df_one['prod'],
                          labels={'y':'Solar Energy Production, W',
                                  'x':'Month'},
                          height=700,
                          )
        return energy_month_fig
        
# Year freq graph    
    elif freq_val=='year_freq':
        df_one = energy_year
        energy_year_fig = px.bar(x=df_one.index, 
                         y=df_one['prod'],
                         labels={'y':'Solar Energy Production, W',
                                 'x':'Year'},
                        #  color=df['prod'],
                         height=700,
                         )
        return energy_year_fig

    else:
        raise dash.exceptions.PreventUpdate

#---------------------------------------------------------------------------------------------
### Download first csv data call
@app.callback(
    Output('download-dataframe-csv-one', 'data'),
    Input('btn_one_csv', 'n_clicks'),
    prevent_initial_call = True,
)
def function(n_clicks):
    return dcc.send_data_frame(df_one.to_csv, f'{df_one.index[0]}.csv')


#---------------------------------------------------------------------------------------------
### Callback for secondary chart

@app.callback(
    Output('secondary-graph-one-update', 'figure'),
    Output('secondary-graph-two-update', 'figure'),
    Output('average-prod', 'children'),
    Output('max-prod', 'children'),
    Output('sum-prod', 'children'),
    Output('comparison-prod', 'children'),
 

    [Input('main-figure-update', 'clickData'),
     Input('apply-consumption-change-button', 'n_clicks'),
     
     State('input-consumption', 'value'),
     State('radio-items-resample-frequency', 'value')],
)

def sec_graph_fig(clicked_fig, clicked_btn, consumption_input, freq_val):

    # Initial click_btn value is zero, and later we will check whether it is clicked or not
    # current_btn_val = 0
    global date_val_two

    if clicked_fig is None:
        date_val_two = '2021-05-26'
    else:
        date_val_two = clicked_fig['points'][0]['x']

    if freq_val == 'day_freq':
        df_two = energy_hour.loc[date_val_two]
        
        x= 'Hour'
        
        info_date = date_val_two
        # Checking the day of week of specified date_val
        if df_two.index[0].strftime('%w') in range(1,6):
            consumption_y_day = consumption_weekday
        else:
            consumption_y_day = consumption_weekend
        


    elif freq_val == 'month_freq':
        df_two = energy_day.loc[date_val_two[:7]]
        
        x= 'Days of Month'
        
        info_date = date_val_two[:7]

    elif freq_val == 'year_freq':
        df_two = energy_month.loc[date_val_two[:4]]

        x='Months'
        
        info_date = date_val_two[:4]

    
    

    fig_one = go.Figure()
    fig_one.add_trace(go.Bar(
                            x=df_two.index, 
                            y=df_two['prod'],
                            text=df_two['prod'],
                            name='Production',
                            # marker_color='rgb(0, 255, 0)',
                            ))

#---------------------------------------------------------------------------------------------
### First right figure
    fig_one.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_one.update_layout(
                          hovermode='x',
                          uniformtext_minsize=8,
                          uniformtext_mode='hide',
                        #   xaxis_tickangle=45,
                          yaxis=dict(
                              title='Solar Energy Production, W',
                              titlefont_size=16,
                              tickfont_size=14,
                          ),
                          xaxis=dict(
                              title=x,
                              titlefont_size=16,
                              tickfont_size=10,
                          )
                         )

#---------------------------------------------------------------------------------------------
### Second right figure
    
    # global clk_btn_y_df
    comparison = 0
    
    fig_two = go.Figure()
    fig_two.add_trace(go.Bar(
                        x=df_two.index, 
                        y=df_two['prod'],
                        text=df_two['prod'],
                        name='Production',
                        # marker_color='rgb(0, 255, 0)',
                        ))

    if clicked_btn > 0:
        if freq_val == 'day_freq':

            clk_btn_y_df = df_two.reset_index()['prod'] - consumption_y_day[df_two.index[0].strftime('%B')]*consumption_input
            
            comparison = round(clk_btn_y_df.sum(), 2)
            
            fig_one.add_trace(go.Scatter(
                x=df_two.index,
                y=consumption_y_day[df_two.index[0].strftime('%B')]*consumption_input,
                name = 'Consumption',
                mode = 'lines+markers'
                )
            )
            
            fig_two.data = []
            
            fig_two.add_trace(go.Bar(
                    x = df_two.index,
                    y=clk_btn_y_df,
                    name='Comparison'
                ))


# Data

    avg_num = '[{}] average : {} W'.format(info_date , round(df_two['prod'].mean(), 2))
    max_num = '[{}] max : {} W'.format(info_date, round(df_two['prod'].max(), 2))
    sum_num = '[{}] overall : {} W'.format(info_date, round(df_two['prod'].sum(), 2))
    
    if comparison == 0:
        comparison_num = None
    elif comparison > 0:
        comparison_num = 'On [{}] : {} W, Produced more than Consumed'.format(info_date, comparison)
    elif comparison < 0:
        comparison_num = 'On [{}] : {} W, Consumed more than Produced'.format(info_date, comparison*(-1))

    

# Return to Ouput of the Callback

    return fig_one, fig_two, avg_num, max_num, sum_num, comparison_num


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# Download second csv data call
@app.callback(
    Output('download-dataframe-csv-two', 'data'),
    Input('btn_two_csv', 'n_clicks'),
    prevent_initial_call = True,
)
def function(n_clicks):
    return dcc.send_data_frame(energy.loc[date_val_two].to_csv, f'{energy.loc[date_val_two].index[0]}.csv')

#---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)