# -*-coding:utf-8-*-
import time
import sys
import json
import traceback

from jieba_model import JiebaModel

from flask import Flask, request, Response, abort
from flask_cors import CORS  # 解决跨域问题

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['POST', 'GET'])
def index():
    return '分词程序正在运行'

@app.route('/split_words', methods = ['POST', 'GET'])
def get_result():
    if request.method == 'POST':
        text = request.data.decode('utf-8')
    else:
        text = request.args['text']

    try:
        start = time.time()
        print(f'用户输入: {text}')
        res = jiebaModel.generate_result(text)
        end = time.time()
        print(f'分词耗时: {end - start}')
        print(f'分词结果: res')
        result = {'code': '200', 'msg': '响应成功', 'data': res}
    except Exception as e:
        print(e)
        result_error = {'errcode': -1}
        result = json.dumps(result_error, indent = 4, ensure_ascii = False)
        # 这里用于捕获更详细的异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        # 提前退出请求
        abort(Response('Failed!' + ''.join('' + line for line in lines)))

    return Response(str(result), mimetype = 'application/json')

if __name__ == '__main__':
    jiebaModel = JiebaModel()
    jiebaModel.load_model()
    app.run(host = '0.0.0.0', port = 1314, threaded = False)