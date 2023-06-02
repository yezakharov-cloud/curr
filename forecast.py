from flask import Flask, request, render_template_string

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['rate']
        current_rate = f'{user_input:.6f}'  # Adds six zeros after the decimal point
        return f"The current rate is: {current_rate}"
    return render_template_string('''
        <form method="POST" action="/">
            <label for="rate">Enter the rate:</label>
            <input type="number" step="0.000001" name="rate" id="rate" required>
            <input type="submit" value="Submit">
        </form>
    ''')

if __name__ == '__main__':
    app.run()
