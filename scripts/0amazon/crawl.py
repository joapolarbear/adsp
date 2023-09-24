#-*-coding:utf-8-*-

import pdfkit
import requests
import sys
import os
import urllib2
import re


reload(sys)
sys.setdefaultencoding('utf-8')

class Spider:
    def get_pdf(self, urls, name):
        if os.path.exists('E:/linux_pdf'):
            print 'already exists!!!'
        else:
            os.mkdir('E:/linux_pdf')
        os.chdir('E:/linux_pdf/')
        print 'url = ' + urls
        # names = unicode(name, encoding='utf-8')
        # print names

        pdfkit.from_url(urls, name + '.pdf')
        print 'hahhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    def get_main_page(self,url):
        header = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}

        html = requests.get(url)
        url_field = re.findall('<ul class="dropdown-menu">(.*?)</ul>', html.text, re.S)[0]
        url_lists = re.findall('class="menu-item.*?<a href="(.*?)">', url_field, re.S)
        print url_lists
        print '-'*100
        name_lists = re.findall('<a href=".*?">(.*?)</a></li>',url_field, re.S)


        if len(url_lists) == len(name_lists):
            for i in range(1, len(url_lists)+1):
                self.get_pdf(url_lists[i-1], str(i))
        else:
            print "crawl lists is wrong!!!"

if __name__ == '__main__':
    url = 'http://m.one-piece.cn/post/10918/?weixin >> one-piece918.pdf'
    spider = Spider()
    spider.get_main_page(url)