################################ Training ################################

echo 'First arg:' $1

# Duke Dataset
python test.py \
--name pre2concat_512p_train \
--model pix2pixHD \
--batchSize 8 \
--loadSize 512 \
--label_nc 0 \
--input_nc 3 \
--output_nc 3 \
--no_instance \
--dataroot ../../../data/ldm_data_all_phases_concat/test \
--resize_or_crop resize_and_crop \
--how_many 1000000000 \
--which_epoch $1 \
--phase test
#--gpu_ids 4 \
#--tf_log \
#--print_freq 100 \
#--nThreads 0 \
#--gpu_ids -1 \
#--fp16 \
