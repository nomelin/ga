from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, World!你好，世界！"})


if __name__ == '__main__':
    app.run(debug=True)
