#! /bin/bash

#sleep 2h

#### Preliminaries


#echo "1. Activating virtual environment called generative_breast_controlnet_env."
##python3 -m venv MMG_env
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

#git clone https://github.com/Project-MONAI/GenerativeModels.git
#cd GenerativeModels/
#python setup.py install
#cd ..

echo "3. METRIC computation on TEST data."
echo "==========================================================================="
echo "======================== FULL IMAGE ========================"
echo "==========================================================================="

#echo "======================== REAL-SYNTHETIC Comparisons ========================"

# 30
#echo "postcontrast phase 1 real - postcontrast phase 1 synthetic normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1/ /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase1/ --phase 0001 --secondphase 0000 --normalize_images

#echo "postcontrast phase 2 real - postcontrast phase 2 synthetic normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase2 --phase 0002 --secondphase 0000 --normalize_images

#echo "postcontrast phase 3 real - postcontrast phase 3 synthetic normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/pre_post_synthesis-main/synthesis/pix2pixHD/results/pre2concat_512p_train/test_30/phase3 --phase 0003 --secondphase 0000 --normalize_images


echo "======================== REAL-REAL Comparisons ========================"

# 30
#echo "precontrast real - postcontrast phase 1 real normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase1/ /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A --phase 0001 --secondphase 0000 --normalize_images

#echo "precontrast real - postcontrast phase 2 real normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase2 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A --phase 0002 --secondphase 0000 --normalize_images

#echo "precontrast real - postcontrast phase 3 real normalized Pix2PixHD"
#python3 metrics.py /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/phase3 /home/richard/Desktop/software/data/ldm_data_all_phases_concat/test/test_A --phase 0003 --secondphase 0000 --normalize_images



echo "======================== METRICS: REAL-SYNTHETIC Subtraction (GAN) Comparisons IMAGENET ========================"

echo "subtraction phase 1 real - GAN subtraction phase 1 syn normalized imagenet"
python3 metrics.py /home/roo/Desktop/jmi2/test_subtraction/phase_1/targets/  /home/roo/Desktop/jmi2/gan_output_subtracted/phase1/  --phase 0000 --secondphase 0000 --normalize_images
echo "-----"

echo "subtraction phase 2 real - GAN subtraction phase 2 syn normalized imagenet"
python3 metrics.py /home/roo/Desktop/jmi2/test_subtraction/phase_2/targets/  /home/roo/Desktop/jmi2/gan_output_subtracted/phase2/  --phase 0001 --secondphase 0000 --normalize_images
echo "-----"

echo "subtraction phase 3 real - GAN subtraction phase 3 syn normalized imagenet"
python3 metrics.py /home/roo/Desktop/jmi2/test_subtraction/phase_3/targets/  /home/roo/Desktop/jmi2/gan_output_subtracted/phase3/  --phase 0002 --secondphase 0000 --normalize_images
echo "-----"






echo "FINISHED CALCULATING METRICS"
