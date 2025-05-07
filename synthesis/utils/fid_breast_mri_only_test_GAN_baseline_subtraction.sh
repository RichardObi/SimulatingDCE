#! /bin/bash

#sleep 2h
#### Preliminaries

#echo "1. Activating virtual environment called generative_breast_controlnet_env."
#python3 -m venv MMG_env
#source ../generative_breast_controlnet_env/bin/activate

#echo "2. Pip install dependencies"
#pip3 install --upgrade pip --quiet
#pip3 install keras
#pip3 install wget --quiet
#pip3 install numpy --quiet
#pip3 install opencv-contrib-python --quiet
#pip3 install opencv-python==4.5.5.64
#pip3 install torchmetrics
#pip install numpy==1.21
#pip install torchmetrics[image]

#conda install -c conda-forge cudnn=7.6.5=cuda10.1_0 cudatoolkit=11.2

#conda install -c conda-forge cudatoolkit=11.2 cudnn=7.6.5=cuda10.1_0

echo "Now installing tensorflow using conda install tensorflow-gpu - please stop (crtl-c) process now if you would not like to install tf in your environment."
sleep 10s
conda install tensorflow-gpu

#export CUDA_VISIBLE_DEVICES=0

echo "3. FID computation on TEST DATASET: Start"

#echo "==========================================================================="
#echo "======================== FULL IMAGE ========================"
#echo "==========================================================================="

echo "==================== IMAGENET: ===================="

echo "======================== REAL-SYNTHETIC Subtraction (U-NET) Comparisons IMAGENET ========================"

echo "subtraction phase 1 real - U-NET subtraction phase 1 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase1 --phase 0001 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_p1_vs_syn_p1_imagenet_normalized

echo "subtraction phase 2 real - U-NET subtraction phase 2 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase2 --phase 0002 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_p2_vs_syn_p2_imagenet_normalized

echo "subtraction phase 3 real - U-NET subtraction phase 3 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase3 --phase 0003 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_p3_vs_syn_p3_imagenet_normalized



echo "======================== REAL Pre - REAL Subtraction Comparisons IMAGENET ========================"

echo "precontrast real - subtraction phase 1 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A --phase 0001 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_real_p1_imagenet_normalized

echo "precontrast real - subtraction phase 2 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A  --phase 0002 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_real_p2_imagenet_normalized

echo "precontrast real - subtraction phase 3 syn normalized imagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A  --phase 0003 --secondphase 0000 --normalize_images --limit 99999999 --model imagenet --description real_real_p3_imagenet_normalized



echo "==================== RADIMAGENET: ===================="

echo "======================== REAL-SYNTHETIC Subtraction (U-NET) Comparisons radimagenet ========================"

echo "subtraction phase 1 real - U-NET subtraction phase 1 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase1 --phase 0001 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_p1_vs_syn_p1_radimagenet_normalized

echo "subtraction phase 2 real - U-NET subtraction phase 2 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase2 --phase 0002 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_p2_vs_syn_p2_radimagenet_normalized

echo "subtraction phase 3 real - U-NET subtraction phase 3 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase3 --phase 0003 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_p3_vs_syn_p3_radimagenet_normalized


echo "======================== REAL Pre - REAL Subtraction Comparisons radimagenet ========================"

echo "precontrast real - subtraction phase 1 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A --phase 0001 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_real_p1_radimagenet_normalized

echo "precontrast real - subtraction phase 2 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A  --phase 0002 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_real_p2_radimagenet_normalized

echo "precontrast real - subtraction phase 3 syn normalized radimagenet"
python3 fid.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A  --phase 0003 --secondphase 0000 --normalize_images --limit 99999999 --model radimagenet --description real_real_p3_radimagenet_normalized



echo "3. FID computation on TEST DATASET: Done"

