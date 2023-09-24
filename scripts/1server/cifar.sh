#!/usr/bin/expect 
set timeout 4

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


set user net
set dataset cifar10
set timeout 4

set port_base_usp 14200
set port_base_ssp 14270
set port_base_alter 14250
set port_base_tap 14260
set port_base_ada 14220

set training_end 0.9
set batch_size 128
set class_num 10
set check_period 60.0
set base_time_step 20.0
set s 40

# login
spawn ssh hphu@gatekeeper.cs.hku.hk
expect "*Password: "
send "252915\r"
expect "color)"
send "\r"
expect "exit"
send "ssh $user@$ip\r"
# send "ssh net@net-b1\r"
expect "(yes/no)?"
send "yes\r"
expect "$user@"
send "netexplo\r"


# run
expect "$user@"
send "cd hphu/$dataset/\r"

if { $job_name == "ps" } {
	expect "$user@"
	send "rm ps* \r"

	expect "$user@"
	send "nohup python2.7 -u client.py \
		--job_name=ps \
		--host='$ps_ip' \
		--port_base=$port_base_usp \
		--base_dir='/home/$user/hphu/$dataset/' \
		--class_num=$class_num \
		--worker_num=$wk_num \
		--check_period=$check_period \
		--s=$s \
		--training_end=$training_end > ps_nohup.txt 2>&1 &\r"
} elseif { $job_name == "worker" } {
	expect "$user@"
	send "rm wk_* \r"

	if { $wk_index >= [expr {$wk_num - 1}] } {
		expect "$user@"
		send "nohup python2.7 -u client.py \
			--job_name=worker \
			--worker_index=$wk_index  \
			--sleep_time=1 \
			--host='$ps_ip' \
			--port_base=$port_base_usp \
			--base_dir='/home/$user/hphu/$dataset/' \
			--base_time_step=$base_time_step \
			--batch_size=$batch_size \
			--check_period=$check_period \
			--s=$s \
			--class_num=$class_num > wk_'$wk_index'_nohup.txt 2>&1 &\r"
	} else {
		expect "$user@"
		send "nohup python2.7 -u client.py \
			--job_name=worker \
			--worker_index=$wk_index  \
			--sleep_time=0 \
			--host='$ps_ip' \
			--port_base=$port_base_usp \
			--base_dir='/home/$user/hphu/$dataset/' \
			--base_time_step=$base_time_step \
			--batch_size=$batch_size \
			--check_period=$check_period \
			--s=$s \
			--class_num=$class_num > wk_'$wk_index'_nohup.txt 2>&1 &\r"
	}

}





# exit
expect "$user@"
send "exit\r"
expect eof
send "exit\r"
expect eof

