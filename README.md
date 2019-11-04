# STrain

Currently, it needs users to manually lauch the parameter server first and then all workers.

First, to lanch the parameter server, you can run the following cammands on the server where you want to run parameter server.

```bash
sh run.sh 'usp' 'ps'
```
More detaily, you can also set the arguments yourself.

```bash
python2.7 -u client.py \
			--job_name=ps \
			--host=$ps_ip \
			--port_base=14200 \
			--base_dir="/home/$usr_name/hphu/cifar10/" \
			--class_num=10 \
			--worker_num=$worker_num \
			--check_period=$check_period \
			--training_end=$training_end
```

Then for each worker, you can run the following commands.
```bash
sh run.sh 'usp' 'wk'
```
More detaily, you can also set the arguments yourself.

```bash
python2.7 -u client.py \
			--job_name=worker \
			--worker_index=$wk_index  \
			--sleep_time=$sleep_time \
			--host=$ps_ip \
			--port_base=14200 \
			--base_dir="/home/$usr_name/hphu/cifar10/" \
			--batch_size=$batch_size \
			--check_period=$check_period \
			--class_num=10
```


