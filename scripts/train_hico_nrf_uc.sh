CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 25623 \
        --use_env \
        main.py \
        --output_dir output/nrf_uc \
        --dataset_file hico \
        --hoi_path ./data/hico_20160224_det/ \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --fix_clip \
        --batch_size 20 \
        --pretrained ./params/hico_50_4_noattention.pth \
        --with_clip_label \
        --gradient_accumulation_steps 1 \
        --num_workers 8 \
        --opt_sched "multiStep" \
        --dataset_root GEN \
        --model_name HOICLIP \
        --del_unseen \
        --zero_shot_type non_rare_first \
        --resume ${EXP_DIR}/checkpoint_best.pth \
        --verb_pth ./tmp/verb.pth \
        --training_free_enhancement_path \
        ./training_free_ehnahcement/

