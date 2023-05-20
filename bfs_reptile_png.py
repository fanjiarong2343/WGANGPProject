import requests
from bs4 import BeautifulSoup
import re
from sequence.utils import *
from config import *

# 已访问过的URL集合
url_set = {}
format_dir = 'data/libpng/format_set'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'}


def is_visit(url_set, url):
    if url_set.get(url) is None:
        return False
    else:
        return True


# 单个img下载
def download_img(url, i):
    try:
        response = requests.get(url)
        # 获取的文本实际上是图片的二进制文本
        img = response.content

        # 保存路径 wb代表写入二进制文本
        path = '{}/org_png:id:{}'.format(format_dir, str(i).zfill(4))
        with open(path, 'wb') as f:
            f.write(img)
        f.close()
    except Exception as e:
        print(e)


def neighbor(a_items):
    neighbor_url = []
    for a in a_items:
        href = a.get("href")
        if href is None:
            continue
        if href[0] == '/':
            url = "https://pngpai.com" + href
            if not is_visit(url_set, url):
                neighbor_url.append(url)
    return neighbor_url


# 批量img下载
def png(queue, url, i):
    c = requests.get(url, headers=headers, timeout=30)
    c.encoding = "utf-8"
    html = c.text
    soup = BeautifulSoup(html, features="html.parser")
    url_set[url] = 1

    a_items = soup.find_all("a")
    neighbor_url = neighbor(a_items)
    for n_url in neighbor_url:
        # 如果n_url没访问过
        if not is_visit(url_set, n_url):
            queue.append(n_url)

    img_items = soup.find_all("img")
    for item in img_items:
        src = item.get('src')
        if src is None:
            continue
        if src[0] == '/':
            img_url = "https://pngpai.com" + src
        print("img_url:", img_url)
        if img_url.find('.png') != -1:
            download_img(img_url, i)
        i += 1
    return i


# 输入一个起始点 URL
def BFS(url, i):
    # 创建队列
    queue = []
    queue.append(url)

    # 当队列不空的时候
    while len(queue) > 0:
        # 将队列的第一个元素读出来
        url = queue.pop(0)
        print("url:", url)
        new_i = png(queue, url, i)
        if new_i > i:
            i = new_i
        if i > 2000:
            break


if __name__ == '__main__':
    url = "https://pngpai.com/"
    i = 1
    BFS(url, i)