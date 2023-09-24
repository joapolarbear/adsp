#!/usr/bin/expect 
# set the user name of ssh connection
set user ubuntu

 # string
set ip [lindex $argv 0]
# string
set ps_ip [lindex $argv 1] 
# integer
set wk_num [lindex $argv 2] 
# integer
set wk_index [lindex $argv 3] 
# string
set job_name [lindex $argv 4]  

set model [lindex $argv 5]

set port_base_usp 14000
set port_base_ssp 14270
set port_base_alter 14250
set port_base_tap 14260
set port_base_ada 14220
set timeout 180
set training_end 1.0
set batch_size 128

set check_period 60.0
set s 40

set sleep_time 0



####################################
spawn ssh -i "strain.pem" $user@$ip

# expect "$user@"
# send "mkdir hphu/\r"


###########################################################################
### run cifar10
expect "$user@"
send "cd hphu/cifar10/\r"


if { $job_name == "ps" } {
	expect "$user@"
	send "source activate tensorflow_p27\r"

	expect "$user@"
	send "rm ps* \r"

	if { $model == "usp" } {
		expect "$user@"
		send "nohup python2.7 -u client.py \
			--job_name=ps \
			--host='$ps_ip' \
			--port_base=$port_base_usp \
			--base_dir='/home/ubuntu/hphu/cifar10/' \
			--class_num=10 \
			--worker_num=$wk_num \
			--check_period=$check_period \
			--epsilon=0.3 \
			--training_end=$training_end > ps_nohup_usp.txt 2>&1 &\r"
	} elseif { $model == "ssp" } {
		expect "$user@"
		send "nohup python2.7 -u ssp_PS.py \
				--host=$ps_ip \
				--port_base=$port_base_ssp \
				--base_dir='/home/ubuntu/hphu/cifar10/' \
				--class_num=10 \
				--worker_num=$wk_num \
				--check_period=$check_period \
				--s=$s \
				--training_end=$training_end > ps_nohup_ssp.txt 2>&1 &\r"
	} elseif { $model == "alter" } {
		expect "$user@"
		send "nohup python2.7 -u alter_ssp_PS.py \
			--host='$ps_ip' \
			--port_base=$port_base_alter \
			--base_dir='/home/ubuntu/hphu/cifar10/' \
			--class_num=10 \
			--worker_num=$wk_num \
			--check_period=$check_period \
			--s=$s \
			--training_end=$training_end > ps_nohup_alter_ssp.txt 2>&1 &\r"
	} elseif { $model == "ada" } {
		expect "$user@"
		send "nohup python2.7 -u adacomm_PS.py \
			--host=$ps_ip \
			--port_base=$port_base_ada \
			--base_dir='/home/ubuntu/hphu/cifar10/' \
			--class_num=10 \
			--worker_num=$wk_num \
			--check_period=$check_period \
			--training_end=$training_end > ps_nohup_ada.txt 2>&1 &\r"
	}
} elseif { $job_name == "worker" } {
	expect "$user@"
	send "source activate tensorflow_p27\r"

	expect "$user@"
	send "rm wk_* \r"

	if { $model == "usp" } {
		if { $wk_index > 10} {
			expect "$user@"
			send "nohup python2.7 -u client.py \
				--job_name=worker \
				--worker_index=$wk_index  \
				--sleep_time=$sleep_time \
				--host='$ps_ip' \
				--port_base=$port_base_usp \
				--base_dir='/home/ubuntu/hphu/cifar10/' \
				--batch_size=$batch_size \
				--check_period=$check_period \
				--class_num=10 > wk_'$wk_index'_nohup_usp.txt 2>&1 &\r"
		} else {
			expect "$user@"
			send "nohup python2.7 -u client.py \
				--job_name=worker \
				--worker_index=$wk_index  \
				--sleep_time=0 \
				--host='$ps_ip' \
				--port_base=$port_base_usp \
				--base_dir='/home/ubuntu/hphu/cifar10/' \
				--batch_size=$batch_size \
				--check_period=$check_period \
				--class_num=10 > wk_'$wk_index'_nohup_usp.txt 2>&1 &\r"
		}
	} elseif { $model == "ssp" } {
		expect "$user@"
		send "nohup python2.7 -u ssp_WK.py \
				--worker_index=$wk_index \
				--s=$s \
				--sleep_time=0 \
				--host=$ps_ip \
				--port_base=$port_base_ssp \
				--base_dir='/home/ubuntu/hphu/cifar10/' \
				--batch_size=$batch_size \
				--class_num=10 > wk_'$wk_index'_nohup_ssp.txt 2>&1 &\r"
	} elseif { $model == "alter" } {
		if { $wk_index > 10} {
			expect "$user@"
			send "nohup python2.7 -u alter_ssp_WK.py \
					--worker_index=$wk_index \
					--s=$s \
					--sleep_time=$sleep_time \
					--host=$ps_ip \
					--port_base=$port_base_alter \
					--base_dir='/home/ubuntu/hphu/cifar10/' \
					--batch_size=$batch_size \
					--class_num=10 > wk_'$wk_index'_nohup_alter_ssp.txt 2>&1 &\r"
		} else {
			expect "$user@"
			send "nohup python2.7 -u alter_ssp_WK.py \
					--worker_index=$wk_index \
					--s=$s \
					--sleep_time=0 \
					--host=$ps_ip \
					--port_base=$port_base_alter \
					--base_dir='/home/ubuntu/hphu/cifar10/' \
					--batch_size=$batch_size \
					--class_num=10 > wk_'$wk_index'_nohup_alter_ssp.txt 2>&1 &\r"
		}
	} elseif { $model == "ada" } {
		expect "$user@"
		send "nohup python2.7 -u adacomm_WK.py \
				--worker_index=$wk_index \
				--sleep_time=0 \
				--host=$ps_ip \
				--port_base=$port_base_ada \
				--base_dir='/home/ubuntu/hphu/cifar10/' \
				--batch_size=$batch_size \
				--class_num=10 > wk_'$wk_index'_nohup_ada.txt 2>&1 &\r"
	}
	
}
###########################################################################


###########################################################################





# expect '*Password: '
# send "$passwd\r"

# expect {
# 	"(yes/no)?" { send "yes\r"; exp_continue}
# 	"*Password: " { send "$passwd\r"}}
# interact





# expect "$user@"
# send "cd hphu/alter_ssp_cifar10/\r"
# expect "$user@"
# send "mkdir usp_cifar10\r"
# expect "$user@"
# send "source activate tensorflow_p27\r"


## run tensorflow code
# expect "$user@"
# send "source activate tensorflow_p27\r"
# expect "$user@"
# send "python2.7 alter_ssp_WK.py --worker_index=0 --s=40 --sleep_time=0.4 --host='13.59.157.164' --port_base=14300 --base_dir='/home/ubuntu/hphu/alter_ssp_cifar10/' --class_num=10 &\r"

## generate imbalanced data
# expect "$user@"
# send "cd hphu/\r"
# expect "$user@"
# send "source activate tensorflow_p27\r"
# expect "$user@"
# send "python _generate_imbalanced.py 0\r"


# expect "$user@"
# send "ping $ps_ip\r"
# expect "$user@"
# exit

################################
# expect "$user@"
# send "exit"
# interact
####################################
expect "$user@"
send "exit\r"
expect eof  



# ssh -i "strain.pem" ubuntu@18.191.166.129

# scp -r -i "strain.pem" ../Documents/program/python/STrain/* ubuntu@13.59.157.164:~/hphu/

# scp -i "strain.pem" strain.pem $user@$ip:~/hphu/

# source activate tensorflow_p27
# python2.7 alter_ssp_WK.py --worker_index=0 --s=40 --sleep_time=0.4 --host='13.59.157.164' --port_base=14300 --base_dir='/home/ubuntu/hphu/alter_ssp_cifar10/' --class_num=10 &

# python2.7 alter_ssp_PS.py --host='172.31.40.87' --port_base=14300 --base_dir='/home/ubuntu/hphu/alter_ssp_cifar10/' --class_num=10 --check_period=60.0 --training_end=0 &


