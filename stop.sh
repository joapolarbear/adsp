#!/bin/bash
#stop.sh

if [ ! $1 ]; then
	echo "Usage: $0 <name> [l]"
	echo " l just list"
	else
	param=$1
fi

if [ "$2" = "l" -o "$2" = "L" ]; then
	ps aux | grep $1 | grep -v "grep"
	exit
fi

# process every prog like $1

for pid in `ps aux | grep $param | grep -v "grep" | awk '{print $2}'` ; do
	ok=true
	pid_name=` ps aux | grep $pid | sed -n '1P' | awk '{print $11}'`
	#sed -n '1P' 相当于 grep -v grep
	# if have such pid,(becaus some time when the first be killed , the forked pid will be killed all the same, for example mozilla.)
	if [ $pid_name ]; then
		echo -n "name: $pid_name pid: $pid Kill? [Y/n/q]:"
		read confirm
		#want quit...
		if [ "$confirm" = "q" -o "$confirm" = "Q" ]; then
			echo " Quit..."
			exit
		fi
		#echo "$confirm"
		#want to kill..
		if [ "$confirm" = "" ]; then
			echo -n " killing...default"
			kill -9 $pid
			echo "ok."
		else
		if [ $confirm = 'n' -o $confirm = 'N' ]; then
			echo -n " Canceled..."
			echo $'a'
			continue
		else
		if [ "$confirm" = "y" -o "$confirm" = "Y" ]; then
			echo -n " killing...(confirmed)!"
			kill -9 $pid
			echo "ok."
		fi
	fi
	fi
	# no such pid
	else
	echo "no such pid $pid";
	continue
	fi

done

# kill over. 
if [ $ok ]; then
	echo "All prog like "$1" were prcessed by you!"
else
	echo "Sorry, No such prog like "$1""
fi

# -------------------------------------------------------------------------------------

# 使用方法：杀掉进程

# sh stop.sh gbased
# name: /usr/bin/mozilla pid: 26608 Kill? [Y/n/q]
# 默认为Y,输入时大小定皆可,
# 直接回车为杀,输Y或y也杀.
# 输n或N为不杀,并继续下一个.
# 输q或Q为退出脚本.

# ----------------------------

# 查看进程

# sh stop.sh gbased L

# zhouyan   1833  0.0  0.0 1174556 8924 ?  Sl Oct13 0:00 libexec/gbased --defaults-file=etc/gs_express.cnf

# -------------------------------------------------------------------------------------------

