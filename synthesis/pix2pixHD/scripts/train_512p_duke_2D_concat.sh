################################ Training ################################

# Duke Dataset
python train.py \
--name pre2concat_512p_train \
--model pix2pixHD \
--batchSize 8 \
--loadSize 512 \
--label_nc 0 \
--input_nc 3 \
--output_nc 3 \
--no_instance \
--dataroot ../../../data/ldm_data_all_phases_concat/train \
--save_epoch_freq 5 \
--resize_or_crop resize_and_crop \
--continue_train
#--gpu_ids 4 \
#--tf_log \
#--print_freq 100 \
#--nThreads 0 \
#--gpu_ids -1 \
#--fp16 \
