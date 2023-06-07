# STrain

STrain uses a novel parameter synchronization model, namely ADSP (ADaptive Synchronous Parallel), for distributed machine learning (ML) with heterogeneous devices. It eliminates the significant waiting time occurring with previous parameter synchronization models. The core idea of ADSP is to let faster edge devices continue training, while committing their model updates at strategically decided intervals. We design algorithms that decide time points for each worker to commit its model update, and ensure not only global model convergence but also faster convergence.
 
## Usage

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
## Research Paper
STrain has been accepted by AAAI'20 ([Link to the paper](https://i.cs.hku.hk/~cwu/papers/hphu-aaai20.pdf)). If you use STrain in your research, please consider citing our paper:
```
@inproceedings{hu2020distributed,
  title={Distributed machine learning through heterogeneous edge systems},
  author={Hu, Hanpeng and Wang, Dan and Wu, Chuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={7179--7186},
  year={2020}
}
```

