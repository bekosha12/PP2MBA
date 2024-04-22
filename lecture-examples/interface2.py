import dash_bootstrap_components as dbc
from dash import Input, Output, html
import dash
from dash import html, dcc, callback, Output, Input, State
import pandas as pd

#C:\Users\Admin\Desktop\Case\rating.xlsx
#id -> pd -> rating
#if new -> input age, male + generate new id -> add to the test file
styles = {
    'textarea': {'width': '100%', 'height': 50},
    'button': {'margin-top': '10px'}
}

app = dash.Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])
# Определение макета приложения
content1 = dbc.CardBody(html.Div([html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Div(id = 'container-button-basic',
             children='Enter a value')], className="card-text"))


content2 = dbc.CardBody(html.Div([html.Label("Male/Female"),
        # dcc.Textarea(id='male', style = styles['textarea']),
        dcc.Dropdown(['Male', 'Female'], '', id='male', style= {'width':'40%'}),
        html.Label("Age:"),
        dcc.Textarea(id='age', style = styles['textarea']),
        html.Button('Submit', id='submit-button', n_clicks=0, style = styles['button']),
        html.P(id = 'output-container')
        ], className="card-text")
        )
app.layout = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(content1,label="Найти по ID", tab_id="tab-1"),
                    dbc.Tab(content2,label="Добавить новый ID", tab_id="tab-2"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
    ]
)
@callback(
    Output("output-container", "children"),   
    Output("submit-button", "n_clicks"), 
    [Input("submit-button", "n_clicks"), Input("male", "value"), Input("age", "value")]
)
def on_button_click(n, male, age):
    df = pd.read_pickle('C:/Users/Admin/Desktop/Case/test_temp.pkl')
    df['customer_id'] = df['customer_id'].astype(str)
    fm = ''
    new_age = ''
    cust_index = ''
    if n == 1:
        print(age)
        if male == 'Male':
            fm = 1
        elif male == 'Female':
            fm = 0
        else: 
            return "Choose gender", 0

        if age.isdigit()==False: 
            return "Incorrect age", 0
        new_age = int(age)
        
        for i in range(100000,1000000, 1):
            print(df.tail())
            if i in df['customer_id'].astype(int).unique():
                continue
            else:
                print(df[df['customer_id']==i])
                cust_index = i
                break
        d = {'customer_id': [cust_index], 
             'age': [new_age], 'male':[fm]}
        temp = pd.DataFrame(data=d)
        df = pd.concat([df, temp])
        df.to_pickle('C:/Users/Admin/Desktop/Case/test_temp.pkl')
        return f"Your ID is: {cust_index}", 0
@callback(
    Output('container-button-basic', 'children'),

    Input('input-on-submit', 'value'),
    prevent_initial_call = True
)
def update_output(value):
    df = pd.read_pickle('C:/Users/Admin/Desktop/Case/test_temp.pkl')
    df['customer_id'] = df['customer_id'].astype(str)
    temp = df[df['customer_id'] == str(value)]
    if temp.empty:
        #введенные новые данные внести в базу
        return 'No ID founded'
    else:
        return html.Div([html.P(f'PD: {temp['PD'].sum()}'), html.P(f'Rating: {temp['Rating'].sum()}')]) 


if __name__ == '__main__':
    app.run(debug=False)

# @app.callback(
#     Output("card-content", "children"), [Input("card-tabs", "active_tab")]
# )
# def tab_content(active_tab):
#     return html.Div([html.P("This is tab {}".format(active_tab)),html.P("This is tab {}".format(active_tab))])

if __name__ == '__main__':
    app.run_server(debug=True)