import requests
import os
import urllib.parse
import json
import jsonpath

header = {
    'User-Agent': 'Mozilla/5.0(Macintosh;Inter Mac OS X 10_13_3) AppleWebkit/537.36 (KHTML,like Gecko)'
                  'Chrom/65.0.3325.162 Safari/537.36'
}

# https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E5%8F%AF%E7%88%B1%E5%A4%B4%E5%83%8F&cl=2&word=%E5%8F%AF%E7%88%B1%E5%A4%B4%E5%83%8F&pn=30&rn=30
#https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E7%8C%AB&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word=%E7%8C%AB&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=30&rn=30&gsm=1e&1581575909562=
# url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&word={}&pn={}&rn=30'
url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word={}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={}&rn=30'
# queryWord字段可自行更改，想搜什么写什么
queryWord = '自行车'
queryWords = urllib.parse.quote(queryWord)
word = queryWords
# print(queryWords)
num = 1
for pn in range(0, 2000, 30):
    try:
        urls = url.format(queryWords, word, pn)
        response = requests.get(urls, headers=header).text
        html = json.loads(response)
        photos = jsonpath.jsonpath(html, '$..thumbURL')

    except:
        pass


    # print(html)
    # photos = jsonpath.jsonpath(html,'$..thumbURL')
    # print(photos)
    def mkdir(path):

        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(path)  # mkdir 创建文件时如果路径不存在会创建这个路径
            print
            "---  new folder...  ---"
            print
            "---  OK  ---"

        else:
            print
            "---  There is this folder!  ---"


    path = 'img/%s' % queryWord  # 自行更改存储地
    mkdir(path)
    if type(photos) is not bool:
        for i in photos:
            try:
                a = requests.get(i, headers=header)
                with open('{}/{}.jpg'.format(path, num), 'wb')as f:
                    print("正在下载第%s张图片" % num)
                    f.write(a.content)
                    num += 1
            except:
                pass
    else:
        pn += 1
