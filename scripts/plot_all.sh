#!/bin/bash

export DATA_DIR="/Users/hhp/Desktop/STrainData&Record/resultData/Lab_record"
export DATA_DIR="/Users/hhp/0/adsp_data/Lab_record"

# python3 vary_network_delay.py
# python3 large_model_vgg.py
# python3 compared2R2SP.py
python3 bandwidth_offline_search_hypeparameter.py
# python3 offline_search_commit_rate.py
# python3 waiting_time.py


# "/Users/hhp/Desktop/STrainData&Record/resultData/amazon"
export DATA_DIR="/Users/hhp/0/adsp_data/amazon"
# python3 chiller_rst.py
# python3 cifar_rst.py
# python3 rail_rst.py
# python3 vary_Hete_degree_size.py
# python3 vary_Hete_degree.py
# python3 vary_scale_size.py