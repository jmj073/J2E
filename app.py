from flask import Flask, url_for
from flask.globals import request
from flask import render_template
from werkzeug.utils import redirect
from flask import make_response

from translator_util import load_translator

app = Flask(__name__)
translator = load_translator()


@app.route('/')
def index():
    # text = request.args.get('text', '')
    # return translator(text)
    return render_template('hello.html')

@app.route('/test/', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        try:
            if request.is_json:
                tt = request.get_json()
                res = make_response(translator(tt['text']))
                res.headers["Access-Control-Allow-Origin"] = "*" # "http://192.168.8.103:8080/test/"
                return res
            else:
                return "json이 아님"
        except KeyError as e:
            return 'KeyError남' 
    else:
        text = request.args.get('text', '')
        if text:
            text = translator(text)
        else:
            print('비어있음!')
        return text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)