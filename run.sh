source /usr/local/anaconda3/bin/activate wangru

avail_gpuid=$(python manager_torch.py 3>&2 2>&1 1>&3)
echo $avail_gpuid

if [ "$avail_gpuid" == "" ]; then
    echo "no free memory"
    exit -1
fi

log_name=run_$(date "+%Y%m%d%H%M%S").log

# simcse无监督训练
# CUDA_VISIBLE_DEVICES=$avail_gpuid nohup python run_simcse.py 1>log/$log_name 2>&1 &

# cdtm无监督训练
CUDA_VISIBLE_DEVICES=$avail_gpuid nohup python run_cdtm.py 1>log/$log_name 2>&1 &

