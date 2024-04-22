import dash
from dash import html, dcc, callback, Output, Input, State
import pandas as pd

# Загрузка данных из CSV файла
df = pd.read_csv(r'C:\Users\Admin\Desktop\Case\submission_xgb_test.csv', header = 0)
df['cu']
# Определение стилей
styles = {
    'textarea': {'width': '100%', 'height': 50},
    'button': {'margin-top': '10px'}
}

# Создание экземпляра приложения Dash
app = dash.Dash(__name__)

# Определение макета приложения
app.layout = html.Div([
    html.H1("Customer Data"),
    html.Div([
        html.Label("Male:"),
        dcc.Textarea(id='male', style = styles['textarea']),
        html.Label("Age:"),
        dcc.Textarea(id='age', style = styles['textarea']),
        html.Label("Customer ID:"),
        dcc.Textarea(id='customer_id', style = styles['textarea']),
        html.Button('Submit', id='submit-button', n_clicks=0, style = styles['button']),
        html.Div(id = 'output-container-button')
    ])
])

# Определение callback для обработки данных и вывода результатов
@app.callback(
    Output('output-container-button', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('male', 'value'),
     State('age', 'value'),
     State('customer_id', 'value')]
)
def update_output(n_clicks, male, age, customer_id):
    if n_clicks > 0:
        # Поиск клиента по customer_id
        customer = df[df['Customer ID'] == customer_id]
        if customer.empty:
            # Если клиент не найден, добавляем новые данные в DataFrame и сохраняем в CSV
            new_data = {'Male': male, 'Age': age, 'Customer ID': customer_id}
            df = df.append(new_data, ignore_index = True)
            df.to_csv('your_data.csv', index = False)
            return f"New data for customer with ID {customer_id} added to the CSV file"
        else:
            # Если клиент найден, выводим данные о нем
            return f"Customer found:\n{customer}"

if __name__ == '__main__':
    app.run_server(debug=True)



# Загрузка данных
# train_data = pd.read_csv('C:/Users/Admin/Desktop/Case/train.csv')
# test_data = pd.read_csv('C:/Users/Admin/Desktop/Case/test.csv')



