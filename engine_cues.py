from typing import Iterable
import torch
import util.misc as utils

@torch.no_grad()
def train_one_epoch(model: torch.nn.Module, tokenizer, llama_model, data_loader: Iterable,
                    device: torch.device, epoch: int, args=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    step = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        file_names = [{'filename': i['filename']} for i in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]
        for t, f in zip(targets, file_names):
            t.update(f)
        clip_img = torch.stack([v['clip_inputs'] for v in targets])
        result = model(samples, tokenizer, llama_model, clip_input=clip_img, targets=targets, args=args)
        step += 1

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_hoi(dataset_file, model, tokenizer, llama_model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        clip_img = torch.stack([v['clip_inputs'] for v in targets]).to(device)

        results = model(samples, tokenizer, llama_model, is_training=False, clip_input=clip_img, targets=targets, args=args)
    metric_logger.synchronize_between_processes()

    return results
