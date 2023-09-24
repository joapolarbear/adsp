#!/usr/bin/expect 
# set the user name of ssh connection
set user hphu
set host gatekeeper.cs.hku.hk
set passwd 252915

set pw2 netexplo
set usrname net
set ip [lindex $argv 0]

set timeout 4

spawn ssh $user@$host
# expect '*Password: '
# send "$passwd\r"

expect {
	"(yes/no)?" { send "yes\r"; exp_continue}
	"*Password: " { send "$passwd\r"}}

expect "TERM = (xterm-256color)"
send "\r"

expect "$"
send "ssh $usrname@$ip\r"
expect {
	"(yes/no)?" { 
		send "yes\r";
		expect "*assword: "
		send "$pw2\r";
	 	exp_continue} 
	 "*Password: " { send "$pw2\r";}}
expect "$usrname@"
send "mkdir hphu/\r"
expect "$usrname@"
exit
exit