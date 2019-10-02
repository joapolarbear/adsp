'''
	https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
'''

from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib

'''Get the list of URLs for the images of the synset:'''
page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289")#ship synset
print(page.content)
# BeautifulSoup is an HTML parsing library
soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line


'''Split the urls so each one appears on a different line and store them on a list so they are easy to access:'''
str_soup=str(soup)	#convert soup to string so it can be split
type(str_soup)
split_urls=str_soup.split('\r\n')	#split so each url is a different possition on a list
print(len(split_urls))	#print the length of the list so you know how many urls you have

'''Create directories on the Google Colaboratory file system so the images can be stored there'''
!mkdir /content/train #create the Train folder
!mkdir /content/train/ships #create the ships folder
!mkdir /content/train/bikes #create the bikes folder
!mkdir /content/validation
!mkdir /content/validation/ships #create the ships folder
!mkdir /content/validation/bikes #create the bikes folder