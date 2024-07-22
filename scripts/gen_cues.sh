CUDA_VISIBLE_DEVICES=3 python3 -m torch.distributed.launch \
            --nproc_per_node=1 \
            --master_port 25625 \
            --use_env \
            main_gen_cues.py \
            --output_dir output/default \
            --dataset_file hico \
            --hoi_path ./data/hico_20160224_det/ \
            --num_obj_classes 80 \
            --num_verb_classes 117 \
            --backbone resnet50 \
            --num_queries 64 \
            --dec_layers 6 \
            --epochs 90 \
            --lr_drop 50 \
            --use_nms_filter \
            --fix_clip \
            --batch_size 16 \
            --pretrained ./params/hico_50_4_noattention.pth \
            --with_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 4 \
            --opt_sched "multiStep" \
            --dataset_root cues \
            --model_name GEN_cues \
            --zero_shot_type default \
            --visual_cues posture \
#            --eval \
