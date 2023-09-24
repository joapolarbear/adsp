#!/bin/bash
for((i=927; i<=933; i++ )) do
	wkhtmltopdf http://m.one-piece.cn/post/10$i/?wx=  ~/Desktop/one/one-piece$i.pdf 
done