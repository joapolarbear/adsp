#!/bin/bash
ps_ip="202.45.128.146"
wk_index=$3
worker_num=3
usr_name="net"
batch_size=128
sleep_time=0
s=40
training_end=0.7
check_period=60.0

echo $1
if [ "$1" = "usp" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u client.py \
			--job_name=ps \
			--host=$ps_ip \
			--port_base=14200 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--training_end=$training_end > ps_nohup_usp.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u client.py \
			--job_name=worker \
			--worker_index=$wk_index  \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14200 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--batch_size=$batch_size \
			--check_period=$check_period \
			--class_num=10 > wk_"$wk_index"_nohup_usp.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi

elif [ "$1" = "tap" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u tap_PS.py \
			--host=$ps_ip \
			--port_base=14260 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--s=$s \
			--training_end=$training_end > ps_nohup_tap.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u tap_WK.py \
			--worker_index=$wk_index \
			--s=$s \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14260 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_tap.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "ssp" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u ssp_PS.py \
			--host=$ps_ip \
			--port_base=14270 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--s=$s \
			--training_end=$training_end > ps_nohup_ssp.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u ssp_WK.py \
			--worker_index=$wk_index \
			--s=$s \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14270 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_ssp.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "ada" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u adacomm_PS.py \
			--host=$ps_ip \
			--port_base=14220 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--training_end=$training_end > ps_nohup_ada.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u adacomm_WK.py \
			--worker_index=$wk_index \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14220 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_ada.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
elif [ "$1" = "alter" ]; then
	if [ "$2" = "ps" ]; then
		nohup python2.7 -u adacomm_PS.py \
			--host=$ps_ip \
			--port_base=14250 \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--s=$s \
			--Fixed=True \
			--training_end=$training_end > ps_nohup_ada.txt 2>&1 &
	elif [ "$2" = "wk" ]; then
		nohup python2.7 -u adacomm_WK.py \
			--worker_index=$wk_index \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14250 \
			--s=$s \
			--base_dir="/home/$usr_name/hphu/imageNet/" \
			--batch_size=$batch_size \
			--class_num=10 > wk_"$wk_index"_nohup_ada.txt 2>&1 &
	else
		echo "Argument Error!: unexpected '$2'"
	fi
else
	echo "Argument Error!: unexpected '$1'"
fi


# elif [ "$1" = "alter" ]; then
# 	if [ "$2" = "ps" ]; then
# 		nohup python2.7 -u alter_ssp_PS.py \
# 			--host=$ps_ip \
# 			--port_base=14250 \
# 			--base_dir="/home/$usr_name/hphu/imageNet/" \
# 			--class_num=10 \
# 			--worker_num=$worker_num \
# 			--check_period=$check_period \
# 			--s=$s \
# 			--training_end=$training_end > ps_nohup_alter_ssp.txt 2>&1 &
# 	elif [ "$2" = "wk" ]; then
# 		nohup python2.7 -u alter_ssp_WK.py \
# 			--worker_index=$wk_index \
# 			--s=$s \
# 			--sleep_time=$sleep_time \
# 			--host=$ps_ip \
# 			--port_base=14250 \
# 			--base_dir="/home/$usr_name/hphu/imageNet/" \
# 			--batch_size=$batch_size \
# 			--class_num=10 > wk_"$wk_index"_nohup_alter_ssp.txt 2>&1 &
# 	else
# 		echo "Argument Error!: unexpected '$2'"
# 	fi
