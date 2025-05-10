
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

echo "======================== REAL-SYNTHETIC Post (GAN) Comparisons ========================"

echo "postcontrast phase 1 real - postcontrast phase 1 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase1/for_frd /home/roo/Desktop/jmi2/pre2concat_512p_train/phase1/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "postcontrast phase 2 real - postcontrast phase 2 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase2/for_frd /home/roo/Desktop/jmi2/pre2concat_512p_train/phase2/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "postcontrast phase 3 real - postcontrast phase 3 syn frd with masks Pix2PixHD"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase3/for_frd /home/roo/Desktop/jmi2/pre2concat_512p_train/phase3/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"


echo "======================== REAL-REAL Comparisons ========================"

# 30
echo "precontrast real - postcontrast phase 1 real frd with masks"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase1/for_frd /home/roo/Desktop/jmi2/test_subtraction/inputs/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "precontrast real - postcontrast phase 2 real frd with masks"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase2/for_frd /home/roo/Desktop/jmi2/test_subtraction/inputs/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "precontrast real - postcontrast phase 3 real frd with masks"
python3 -m frd_score /home/roo/Desktop/jmi2/ldm_data_all_phases/phase3/for_frd /home/roo/Desktop/jmi2/test_subtraction/inputs/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"



echo "======================== REAL-SYNTHETIC Subtraction (U-NET) Comparisons IMAGENET ========================"

echo "subtraction phase 1 real - U-NET subtraction phase 1 syn normalized imagenet"
python3 -m frd_score /home/roo/Desktop/jmi2/test_subtraction/phase_1/targets/for_frd /home/roo/Desktop/jmi2/test_subtraction/phase_1/prediction/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "subtraction phase 2 real - U-NET subtraction phase 2 syn normalized imagenet"
python3 -m frd_score //home/roo/Desktop/jmi2/test_subtraction/phase_2/targets/for_frd  /home/roo/Desktop/jmi2/test_subtraction/phase_2/prediction/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "subtraction phase 3 real - U-NET subtraction phase 3 syn normalized imagenet"
python3 -m frd_score /home/roo/Desktop/jmi2/test_subtraction/phase_3/targets/for_frd  /home/roo/Desktop/jmi2/test_subtraction/phase_3/prediction/for_frd -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"





echo "======================== REAL-SYNTHETIC Subtraction (GAN) Comparisons IMAGENET ========================"

echo "subtraction phase 1 real - GAN subtraction phase 1 syn normalized imagenet"
python3 -m frd_score /home/roo/Desktop/jmi2/test_subtraction/phase_1/targets/for_frd  /home/roo/Desktop/jmi2/gan_output_subtracted/phase1/for_frd  -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "subtraction phase 2 real - GAN subtraction phase 2 syn normalized imagenet"
python3 -m frd_score /home/roo/Desktop/jmi2/test_subtraction/phase_2/targets/for_frd  /home/roo/Desktop/jmi2/gan_output_subtracted/phase2/for_frd  -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"

echo "subtraction phase 3 real - GAN subtraction phase 3 syn normalized imagenet"
python3 -m frd_score /home/roo/Desktop/jmi2/test_subtraction/phase_3/targets/for_frd  /home/roo/Desktop/jmi2/gan_output_subtracted/phase3/for_frd  -m /home/roo/Desktop/jmi2/masks/all /home/roo/Desktop/jmi2/masks/all --norm_across --norm_type zscore --feature_groups glcm glrlm gldm glszm ngtdm --save_features
echo "-----"



echo "4. FRD computation on TEST DATASET: Done"


