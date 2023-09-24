#!/bin/bash
ps_ip=("net-b1" )
wk_ip=("net-b3" "net-b6" "net-b7")

amazon_ps_ip=("18.222.183.92")
amazon_ps_interip=("172.31.41.138")
amazon_ip=( 
	# t3.xlarge
	"18.191.163.113"
	"13.59.70.131"

	# t2.2xlarge
	"18.224.27.4"
	"18.191.6.125"
	"18.188.194.244"
	"18.220.214.115"

	# t2.xlarge
	"18.188.32.246"
	"3.17.72.132"
	"18.218.161.0"
	"13.58.217.42"
	"18.220.230.190"

	# t2.large
	"18.221.172.215"
	"18.216.155.155"
	"18.218.4.24"
	"18.218.252.13"
	"18.216.157.194"
	"18.223.124.68"
	"3.16.89.243"
)
wk_num=${#amazon_ip[@]}
wk_num2=$[2*wk_num]
echo $wk_num2
# wk_num=18


echo $1
if [ "$1" = "test" ]; then
	echo $1
elif [ "$1" = "run" ]; then
	for(( ip=0; ip < ${#ps_ip[@]}; ip++ ))
	do
		/usr/bin/expect run.sh "${ps_ip[$ip]}" > ~/Desktop/log/log_${ps_ip[$ip]}.out
		# echo $ip
	done
	# /usr/bin/expect run.sh "${wk_ip[0]}" > ~/Desktop/log/log_${wk_ip[0]}.out &
	for(( ip=0; ip < ${#wk_ip[@]}; ip++ ))
	do
		/usr/bin/expect run.sh "${wk_ip[$ip]}" > ~/Desktop/log/log_${wk_ip[$ip]}.out &
		# echo $ip
	done
elif [ "$1" = "mv" ]; then
	for(( ip=1; ip < ${#ps_ip[@]}; ip++ ))
	do
		/usr/bin/expect mvFile2allwk.sh "${wk_ip[0]}" "${wk_ip[$ip]}" > ~/Desktop/log/log_${wk_ip[$ip]}.out &
	done
elif [ "$1" = "mkdir" ]; then
	for(( ip=0; ip < ${#ps_ip[@]}; ip++ ))
	do
		/usr/bin/expect mkdir.sh "${wk_ip[$ip]}" > ~/Desktop/log/log_${wk_ip[$ip]}.out &
	done
elif [ "$1" = "record" ]; then
	mkdir /Users/hhp/Desktop/amazon/$2
	scp -r -i "strain.pem" ubuntu@${amazon_ps_ip[0]}:~/hphu/cifar10/ps* /Users/hhp/Desktop/amazon/$2/
	# scp -r -i "strain.pem" ubuntu@${amazon_ps_ip[0]}:~/hphu/alter_ssp_cifar10/ps* ubuntu@${amazon_ps_ip[0]}:~/hphu/alter_ssp_cifar10/nohup.txt /Users/hhp/Desktop/amazon/$2/
elif [ "$1" = "amazon" ]; then
	/usr/bin/expect amazon.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps"
	for(( ip=0; ip < $wk_num; ip++ ))
	do
		/usr/bin/expect amazon.sh "${amazon_ip[$ip]}" "${amazon_ps_ip[0]}" $wk_num $ip "worker" 	
	done
elif [ "$1" = "cifar10" ]; then
	/usr/bin/expect cifar10.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps" "usp"
	for(( ip=0; ip < $wk_num; ip++ ))
	do
		/usr/bin/expect cifar10.sh "${amazon_ip[$ip]}" "${amazon_ps_ip[0]}" $wk_num $ip "worker" "usp"	
	done
elif [ "$1" = "cifar10_2" ]; then
	/usr/bin/expect cifar10.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num2 -1 "ps" "alter"
	for(( ip=0; ip < $wk_num2; ip++ ))
	do
		/usr/bin/expect cifar10.sh "${amazon_ip[$ip / 2]}" "${amazon_ps_ip[0]}" $wk_num2 $ip "worker" "alter"	
	done
elif [ "$1" = "rail" ]; then
	/usr/bin/expect rail.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps" "ssp"
	for(( ip=0; ip < $wk_num; ip++ ))
	do
		/usr/bin/expect rail.sh "${amazon_ip[$ip]}" "${amazon_ps_ip[0]}" $wk_num $ip "worker" "ssp"	
	done
elif [ "$1" = "chiller" ]; then
	/usr/bin/expect chiller.sh "${amazon_ps_ip[0]}" "${amazon_ps_interip[0]}" $wk_num -1 "ps" "alter"
	for(( ip=0; ip < $wk_num; ip++ ))
	do
		/usr/bin/expect chiller.sh "${amazon_ip[$ip]}" "${amazon_ps_ip[0]}" $wk_num $ip "worker" "alter"	
	done
elif [ "$1" = "upload" ]; then
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10 ubuntu@${amazon_ps_ip[0]}:~/hphu/cifar10
	scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10/strain_momentum2.py ubuntu@${amazon_ps_ip[0]}:~/hphu/cifar10/strain_momentum.py
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10/alter_ssp_PS.py ubuntu@${amazon_ps_ip[0]}:~/hphu/cifar10/
	for(( ip=0; ip < $wk_num; ip++ ))
	do	
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10 ubuntu@${amazon_ip[$ip]}:~/hphu/cifar10 
		scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10/strain_momentum2.py ubuntu@${amazon_ip[$ip]}:~/hphu/cifar10/strain_momentum.py
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/cifar10/alter_ssp_WK.py ubuntu@${amazon_ip[$ip]}:~/hphu/cifar10/
	done
elif [ "$1" = "upload_rail" ]; then
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/data/* ubuntu@${amazon_ps_ip[0]}:~/hphu/rail/data/
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/strain.py ubuntu@${amazon_ps_ip[0]}:~/hphu/rail/
	scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/ssp_PS.py ubuntu@${amazon_ps_ip[0]}:~/hphu/rail/
	for(( ip=0; ip < $wk_num; ip++ ))
	do	
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/data/* ubuntu@${amazon_ip[$ip]}:~/hphu/rail/data/ 
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/strain.py ubuntu@${amazon_ip[$ip]}:~/hphu/rail/
		scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/rail/ssp_WK.py ubuntu@${amazon_ip[$ip]}:~/hphu/rail/
	done
elif [ "$1" = "upload_chiller" ]; then
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/stop.sh ubuntu@${amazon_ps_ip[0]}:~/hphu/
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/chiller/strain_chiller.py ubuntu@${amazon_ps_ip[0]}:~/hphu/chiller/
	# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/chiller/ssp_PS.py ubuntu@${amazon_ps_ip[0]}:~/hphu/chiller/
	for(( ip=0; ip < $wk_num; ip++ ))
	do	
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/stop.sh ubuntu@${amazon_ip[$ip]}:~/hphu/
		# scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/chiller/strain_chiller.py ubuntu@${amazon_ip[$ip]}:~/hphu/chiller/
		scp -r -i "strain.pem" ~/Documents/program/python/STrain_no_githup/chiller/adacomm_WK.py ubuntu@${amazon_ip[$ip]}:~/hphu/chiller/
	done
else
	echo "Argument Error!: unexpected '$1'"
fi

echo "over"


# ssh -i strain.pem ubuntu@18.222.183.92

# ssh -i strain.pem ubuntu@13.59.90.49

# ssh -i strain.pem ubuntu@18.223.114.201

