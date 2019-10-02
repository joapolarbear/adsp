#!/bin/bash
ps_ip="202.45.128.146"
wk_index=$3
worker_num=3
usr_name="net"
batch_size=64

training_end=1
check_period=60.0
sleep_time=8

s=40


echo $1
if [ "$1" = "usp" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u client_chiller.py \
			--job_name=ps \
			--host=$ps_ip \
			--port_base=14300 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=60.0 \
			--batch_size=$batch_size \
			--training_end=$training_end > ps_nohup_usp.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u client_chiller.py \
			--job_name=worker \
			--worker_index=$wk_index  \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14300 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_usp.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "ssp" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u ssp_PS.py \
			--host=$ps_ip \
			--port_base=14370 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--s=$s \
			--batch_size=$batch_size \
			--training_end=$training_end > ps_nohup_ssp.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u ssp_WK.py \
			--worker_index=$wk_index \
			--s=$s \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14370 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_ssp.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "ada" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u adacomm_PS.py \
			--host=$ps_ip \
			--port_base=14320 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--batch_size=$batch_size \
			--training_end=$training_end > ps_nohup_ada.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u adacomm_WK.py \
			--worker_index=$wk_index \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14320 \
			--base_dir="/home/$usr_name/hphu/rail/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_ada.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "alter" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u adacomm_PS.py \
			--host=$ps_ip \
			--port_base=14350 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--s=$s \
			--Fixed=True \
			--batch_size=$batch_size \
			--training_end=$training_end > ps_nohup_alter_ssp.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u adacomm_WK.py \
			--worker_index=$wk_index \
			--s=$s \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14350 \
			--base_dir="/home/$usr_name/hphu/chiller/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_alter_ssp.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
else
	echo "Argument Error!: unexpected '$1'"
fi
