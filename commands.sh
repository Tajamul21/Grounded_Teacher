# Source Train and Predictions
    # Activate the <Source Train Environment>
    # Medical Dataset 
    cd Source
    python trainval_pretrain_adv.py --dataset voc_medical_trainval --dataset_t voc_medical --net vgg16 --log_ckpt_name "city_voc" --save_dir "output"

    # python trainval_pretrain_adv.py \
    # --dataset voc_medical_train \
    # --dataset_t voc_medical \
    # --net vgg16 \
    # --log_ckpt_name "DDSMSource" \
    # --save_dir "output"
    python eval.py --dataset_t voc_2007_test --load_name /home/tawheed/Grounded_Teacher/AASFOD/output/vgg16/city_voc/lg_adv_session_1_epoch_6_step_10000.pth
    python city_eval.py --dataset_t voc_2007_test --load_name /home/tawheed/Grounded_Teacher/AASFOD/output/vgg16/city_voc/lg_adv_session_1_epoch_6_step_10000.pth

    python psudo_label_generation.py \
    --dataset_t voc_medical \
    --net vgg16 \
    --log_ckpt_name "Pseudo_BraTS" \
    --save_dir "output" \
    --load_name "/DATA/Tawheed/SFDA/Grounded_Teacher/Source/output/vgg16/city_voc/lg_adv_session_1_epoch_6_step_10000.pth"

    # Natural Dataset 
    python trainval_pretrain_adv.py --dataset cs --net vgg16 --log_ckpt_name "citySource" --save_dir "output"
    python psudo_label_generation.py --dataset_t cs_fg --net vgg16 --log_ckpt_name "PseudoL_city2foggy" --save_dir "output" --load_name "output/vgg16/citySource/lg_adv_session_1_epoch_6_step_10000.pth"


# Expert Predictions
    # Medical Dataset 
    cd Expert
    python prediction.py --root "<DATASET_PATH>"

# GT
    # Activate the <GT Environment>
    # Medical Dataset 
    mkdir -p output/city2foggy

    python train_net.py --num-gpus 1 --config configs/faster_rcnn_VGG_cross_city.yaml OUTPUT_DIR output/city2foggy 

    # calculate the froc 
    python eval.py --setting rsna --root output/rsna
