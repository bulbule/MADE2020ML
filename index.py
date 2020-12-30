from datetime import datetime
from datetime import date

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import pandas as pd

from script_LSTM import full_period, STOCKS_NAMES
from models_and_data.description import DESCRIPTION


from gaussian_process import get_prediction

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df_boeing = pd.read_csv("models_and_data/boeing.csv")
df_cisco = pd.read_csv("models_and_data/cisco.csv")
df_intel = pd.read_csv("models_and_data/intel.csv")


for df in [df_boeing, df_cisco, df_intel]:
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df.set_index('Date', inplace=True)


app.layout = html.Div(children=[
    html.H1(children='Market oracle'),

    dcc.Markdown(children=DESCRIPTION),
    dcc.Dropdown(
        id="tool",
        options=[
            {'label': 'Week predictor with GP', 'value': 'GPy'},
            {'label': 'Automatic trader', 'value': 'NN'}
        ],
        placeholder="Select a tool...",
        style={'width': '40%'}
    ),
    html.Label("Enter cash for Automatic trader:"),
    html.Div([
        dcc.Input(
            id="input", type="number", placeholder="input with range",
            min=1000, max=3000, step=100,
        ),
        ],
        style={'display': 'block'}
    ),
    html.Label('Pick a date:'),
    dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed=date(2017, 1, 1),
        max_date_allowed=date(2017, 12, 31),
        initial_visible_month=date(2017, 8, 5),
        date=date(2017, 1, 19)
    ),
    html.Div(id='load'),
    html.Div(id='output'),
])


@app.callback(
    Output('output', 'children'),
    Input("tool", 'value'),
    Input("date-picker", 'date'),
    Input("input", 'value')
)
def update_fig(tool, date_value, money):
    if tool == None:
        return
    date_object = date.fromisoformat(date_value)
    date_string = date_object.strftime('%Y-%m-%d')
    fig = go.Figure()
    if tool == "GPy":
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=("Boeing", "Cisco", "Intel"),
                            shared_xaxes=True,)
        period = 5
        row = 1
        for df in [df_boeing, df_cisco, df_intel]:
            train_dates_start, test, y_pred, y_std = get_prediction(date_string, df, period)
            y_pred, y_std = y_pred.squeeze(), y_std.squeeze()
            dates_pred = test.index

            fig.add_trace(go.Scatter(x=dates_pred, y=y_pred - y_std,
                                mode='lines',
                                line_color='rgb(209, 167, 242)',
                                showlegend=False,),
                                row=row, col=1)
            fig.add_trace(go.Scatter(x=dates_pred, y=y_pred + y_std,
                                mode='lines',
                                fill="tonexty",
                                name='Variance of prediction',
                                line_color='rgb(209, 167, 242)',
                                showlegend=True if row==1 else False),
                                row=row, col=1)
            fig.add_trace(go.Scatter(x=dates_pred, y=test['Open'],
                                mode='markers',
                                name='Data',
                                line_color='black',
                                showlegend=True if row==1 else False),
                                row=row, col=1)
            fig.add_trace(go.Scatter(x=dates_pred[:-period], y=y_pred[:-period],
                                mode='lines',
                                name='Fitted',
                                line_color='rgb(56, 11, 179)',
                                showlegend=True if row==1 else False),
                                row=row, col=1)

            fig.add_trace(go.Scatter(x=dates_pred[-period:], y=y_pred[-period:],
                                mode='markers',
                                marker_symbol=4,
                                name='Predicted',
                                line_color='indigo',
                                showlegend=True if row==1 else False),
                                row=row, col=1
                          )
            fig.update_xaxes(title_text="Date", row=row, col=1)
            fig.update_yaxes(title_text="Open price", row=row, col=1)
            row += 1
        fig.update_layout(height=700)
    else:
        if money == None:
            return
        final_date = date.fromisoformat(date_value)
        final_cash, dates, true_prices, predicted_prices, wallet_cost = full_period(money, final_date)
        dates = [date_1.date() for date_1 in dates]
        fig = make_subplots(rows=4, cols=1,
                            subplot_titles=("Boeing", "Cisco", "Intel", "Wallet"),
                            shared_xaxes=True,)
        row = 1
        for stock_name in STOCKS_NAMES:
            fig.add_trace(go.Scatter(x=dates,
                                     y=true_prices[stock_name],
                                     line_color='black',
                                     mode='lines+markers',
                                     showlegend=True if row == 1 else False,
                                     name='True prices'),
                          row=row, col=1)
            fig.add_trace(go.Scatter(x=dates,
                                     y=predicted_prices[stock_name],
                                     line_color='indigo',
                                     mode='lines+markers',
                                     marker_symbol=4,
                                     showlegend=True if row == 1 else False,
                                     name='Predicted prices'),
                          row=row, col=1)
            row += 1

        fig.add_trace(go.Scatter(x=dates,
                             y=wallet_cost,
                             line_color='red',
                             mode='lines+markers',
                             name='Wallet cost'),
                  row=row, col=1)
        fig.update_layout(
            height=1000,
            title='Dependency of the prices at each moment',
            yaxis_title='Money (US dollars)')
    return dcc.Graph(
        id='prices_graph',
        style={'width': '80vw',},
        figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
