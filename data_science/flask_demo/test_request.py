import requests

def get_split_word_result(text):
    res = requests.post(f'http://localhost:1314/split_words', data = str(text).encode('utf-8'))
    print(res.text)

get_split_word_result('说明：通过requests发送post请求，请求数据编码成utf-8的格式，最后得到响应，并利用.text得到结果')