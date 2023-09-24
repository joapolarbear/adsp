#!/bin/bash
amazon_ps_ip=("net-b1" )
amazon_ps_interip=("202.45.128.146")
amazon_ip=("net-b3" "net-b6" "net-b7")
wk_num=${#amazon_ip[@]}
# wk_num=3


echo $1
if [ "$1" = "test" ]; then
	echo $1
elif [ "$1" = "imageNet" ]; then
	/usr/bin/expect imageNet.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps"
	for(( id=0; id < $wk_num; id++ ))
	do
		/usr/bin/expect imageNet.sh "${amazon_ip[$id]}" "${amazon_ps_interip[0]}" $wk_num $id "worker" 	
	done
elif [ "$1" = "cifar10" ]; then
	/usr/bin/expect  cifar.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps"
	for(( id=0; id < $wk_num; id++ ))
	do
		/usr/bin/expect cifar.sh "${amazon_ip[$id]}" "${amazon_ps_interip[0]}" $wk_num $id "worker" 	
	done
elif [ "$1" = "record" ]; then
	mkdir /Users/hhp/Desktop/Lab_record/$2
	scp  net@${amazon_ps_ip[0]}:~/hphu/cifar10/ps* /Users/hhp/Desktop/Lab_record/$2/
else
	echo "Argument Error!: unexpected '$1'"
fi
echo "over"



