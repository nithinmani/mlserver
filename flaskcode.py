from flask import Flask, jsonify
from flask_cors import CORS
import lstmsample

from jinja2 import escape
app = Flask(__name__)
CORS(app)


@app.route('/ml')
def your_route_function():
    results = lstmsample.run()
    return jsonify(results)


if __name__ == '__main__':
    app.run(port=5000)
