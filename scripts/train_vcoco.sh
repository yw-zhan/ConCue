CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port 25626 \
        --use_env \
        main.py \
        --output_dir output/HCVC_vcoco \
        --dataset_file vcoco \
        --hoi_path ./data/v-coco/ \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 6 \
        --epochs 100 \
        --lr_drop 50 \
        --use_nms_filter \
        --fix_clip \
        --batch_size 15 \
        --pretrained ./params/vcoco_50_4_noattention.pth \
        --with_clip_label \
        --gradient_accumulation_steps 1 \
        --num_workers 8 \
        --opt_sched "multiStep" \
        --dataset_root GEN \
        --model_name HOICLIP \
        --zero_shot_type default \
        --resume ${EXP_DIR}/checkpoint_best.pth \
        --verb_pth ./tmp/vcoco_verb.pth \
        --verb_weight 0.1 \
        --training_free_enhancement_path \
        ./training_free_ehnahcement/
