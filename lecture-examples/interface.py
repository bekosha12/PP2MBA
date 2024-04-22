# from dash import Dash, dcc, html, Input, Output, State, callback
# import pandas as pd
# app = Dash(__name__)
# df = pd.read_excel(r'C:\Users\Admin\Desktop\PP2\RiskCase\Mishki_Gamma.xlsx', converters = {'customer_id': str})
# print(df)
# app.layout = html.Div([
#     html.Div(dcc.Input(id='input-on-submit', type='text')),
#     html.Button('Submit', id = 'submit-val', n_clicks = 0),
#     html.Div(id = 'container-button-basic',
#              children='Enter a value and press submit')
# ])

# #,style={'margin-right': ''}

# @callback(
#     Output('container-button-basic', 'children'),
#     Input('submit-val', 'n_clicks'),
#     Input('input-on-submit', 'value'),
#     prevent_initial_call = True
# )
# def update_output(n_clicks, value):
#     temp = df[df['customer_id'] == str(value)]
#     if temp.empty:
#         #введенные новые данные внести в базу
#         return 'No ID founded'
#     else:
#         return f'PD: {temp['PD'].sum()}'


# if __name__ == '__main__':
#     app.run(debug=False)

# import dash_bootstrap_components as dbc
# from dash import Input, Output, html
# import dash
# from dash import html, dcc, callback, Output, Input, State
# import pandas as pd
# app = dash.Dash(__name__)
# app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# app.layout = html.Div(
#     [
#         dbc.Button(
#             "Click me", id="example-button", className="me-2", n_clicks=0
#         ),
#         html.Span(id="example-output", style={"verticalAlign": "middle"}),
#     ]
# )


# @callback(
#     Output("example-output", "children"),Output("example-button", "n_clicks"), [Input("example-button", "n_clicks")]
# )
# def on_button_click(n):

#     if n is None:
#         return "Not clicked.", 0
#     else:
#         return f"Clicked {n} times.", 0

# if __name__ == '__main__':
#     app.run(debug=False)
import pandas as pd
df = pd.read_excel('C:/Users/Admin/Desktop/Case/test.xlsx', header = 3)
df.to_pickle('C:/Users/Admin/Desktop/Case/test_temp.pkl')