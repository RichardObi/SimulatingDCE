#! /bin/bash

#sleep 2h
#### Preliminaries

echo "1. Let's start"
#echo "1. Activating virtual environment called generative_breast_controlnet_env."
#python3 -m venv MMG_env
#source ../generative_breast_controlnet_env/bin/activate

echo "2. Pip install frd dependency"

pip3 install frd-score

echo "3. FRD computation on TEST DATASET: Start"

#echo "==================== FRD: ===================="

#echo "======================== REAL-SYNTHETIC Comparisons ========================"

#echo "postcontrast phase 1 real - postcontrast phase 1 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase1 -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features 
echo "-----"

#echo "postcontrast phase 2 real - postcontrast phase 2 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase2 -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

#echo "postcontrast phase 3 real - postcontrast phase 3 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase3 -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"



#echo "======================== REAL-REAL Comparisons ========================"

# 30
#echo "precontrast real - postcontrast phase 1 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

#echo "precontrast real - postcontrast phase 2 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

#echo "precontrast real - postcontrast phase 3 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A -M PATH_TO_MASKS/all_masks PATH_TO_MASKS/all_masks --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"



echo "4. FRD computation on TEST DATASET: Done"


