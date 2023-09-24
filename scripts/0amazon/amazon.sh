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


set port_base_usp 14200
set port_base_ssp 14250
set timeout 180
set training_end 0.9
set batch_size 128



####################################
spawn ssh -i "strain.pem" $user@$ip

# expect "$user@"
# send "mkdir hphu/rail/data/ \r"

# expect "$user@"
# send "mkdir hphu/rail/data/data/ \r"

### first time to connect #####
# expect "(yes/no)?"
# send "yes\r"
# expect "$user@"
# send "mkdir hphu/\r"




# expect '*Password: '
# send "$passwd\r"

# expect {
# 	"(yes/no)?" { send "yes\r";}
# 	"*Password: " { send "$passwd\r"}}
# interact





expect "$user@"
send "cd hphu/cifar10/\r"

# expect "$user@"
# send "rm hphu/rail/data/data/.DS_Store\r"

## run tensorflow code
expect "$user@"
send "source activate tensorflow_p27\r"
expect "$user@"
send "python2.7 alter_ssp_WK.py --worker_index=0 --s=40 --sleep_time=0.4 --host='13.59.157.164' --port_base=14300 --base_dir='/home/ubuntu/hphu/cifar10/' --class_num=10 "
interact

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

# expect "$user@"
# send "sh ../stop.sh python2.7 l\r"

expect "$user@"
send "exit\r"
expect eof
####################################




# ssh -i "strain.pem" ubuntu@18.191.166.129

# scp -r -i "strain.pem" ../Documents/program/python/STrain/* ubuntu@13.59.157.164:~/hphu/

# scp -i "strain.pem" strain.pem $user@$ip:~/hphu/

# source activate tensorflow_p27
# python2.7 alter_ssp_WK.py --worker_index=0 --s=40 --sleep_time=0.4 --host='13.59.157.164' --port_base=14300 --base_dir='/home/ubuntu/hphu/alter_ssp_cifar10/' --class_num=10 &

# python2.7 alter_ssp_PS.py --host='172.31.40.87' --port_base=14300 --base_dir='/home/ubuntu/hphu/alter_ssp_cifar10/' --class_num=10 --check_period=60.0 --training_end=0 &


