# Source Train and Predictions
    # Activate the <Source Train Environment>
    # Medical Dataset 
    cd Source
    python trainval_pretrain_adv.py \
    --dataset voc_medical_trainval \
    --dataset_t voc_medical \
    --net vgg16 \
    --log_ckpt_name "DDSMSource" \
    --save_dir "output"

    # python trainval_pretrain_adv.py \
    # --dataset voc_medical_train \
    # --dataset_t voc_medical \
    # --net vgg16 \
    # --log_ckpt_name "DDSMSource" \
    # --save_dir "output"
    
    python psudo_label_generation.py \
    --dataset_t voc_medical \
    --net vgg16 \
    --log_ckpt_name "PseudoL_ddsm2rsna" \
    --save_dir "output" \
    --load_name "output/vgg16/DDSMSource/lg_adv_session_1_epoch_6_step_10000.pth"

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
    mkdir -p output/ddsm2rsna

    python train_net.py \
    --num-gpus 1 \
    --config configs/faster_rcnn_VGG_cross_city.yaml \
    OUTPUT_DIR output/ddsm2rsna \

    # calculate the froc 
    python eval.py --setting ddsm2rsna --root output/ddsm2rsna
