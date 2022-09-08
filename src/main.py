# encoding=utf-8
import copy
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_args
from dataset.data_helper import SampleDataset
from models.modal_classifiser import SampleModel
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, init_distributed_mode
import time
import torch.nn.functional as F
import torch.distributed as dist
from sklearn import metrics


def cal_loss(prediction, label):
    label = label.squeeze(dim=1)
    loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return loss, accuracy, pred_label_id, label


def validate(model, val_dataloader, device, args):
    model.eval()
    predictions = []
    labels = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='valing...', total=len(val_dataloader)):
            text_input, text_mask = batch['text_input'].to(device), batch['text_mask'].to(device)
            token_type_ids, label = batch['token_type_ids'].to(device), batch['label'].to(device)
            prediction = model(text_input, text_mask, token_type_ids)
            # label转换到cuda
            loss, accuracy, pred_label_id, _ = cal_loss(prediction, label)
            loss = loss.mean()
            predictions.append(pred_label_id)
            labels.append(label.squeeze(1))
            losses.append(loss.cpu().numpy())

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    if args.distributed:
        predictions_tensor_list = [torch.zeros_like(predictions) for _ in range(dist.get_world_size())]
        labels_tensor_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(predictions_tensor_list, predictions)
        dist.all_gather(labels_tensor_list, labels)
        predictions = torch.cat(predictions_tensor_list, dim=0)
        labels = torch.cat(labels_tensor_list, dim=0)

    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    macro_f1 = metrics.f1_score(labels, predictions, labels=list(range(0, 36)), average="macro")
    loss = sum(losses) / len(losses)

    model.train()
    return loss, macro_f1


def train_and_validate(args):
    # 1. load data
    device = torch.device(args.device)
    dataset = SampleDataset(args, args.train_data)
    data = dataset.data
    # 按照label分割数据
    labels = []
    for d in data:
        labels.append(d['label_id'])

    sfolder = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True).split(dataset, labels)
    for i, j in sfolder:
        train_index, val_index = i, j
        break
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    print(f"训练集数量：{len(train_dataset)}, 验证集数量：{len(val_dataset)}")
    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True,
                                  drop_last=True, num_workers=args.num_workers, prefetch_factor=args.prefetch,
                                  shuffle=True
                                  )
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.val_batch_size, pin_memory=True,
                                drop_last=False, num_workers=args.num_workers, prefetch_factor=args.prefetch)

    # 2. build model and optimizers
    model = SampleModel(args)
    model.to(device)

    #------------------------------------#
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        

    t_total = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, t_total)

    # 3. training
    step = 0
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    logging.info(f"start time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    for epoch in range(args.max_epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        for batch in tqdm(train_dataloader, desc=f"{epoch}/{args.max_epochs} training...", total=len(train_dataloader)):
            model.train()
            text_input, text_mask = batch['text_input'].to(device), batch['text_mask'].to(device)
            token_type_ids, label = batch['token_type_ids'].to(device), batch['label'].to(device)
            

            prediction = model(text_input, text_mask, token_type_ids)
            loss, accuracy, _, _ = cal_loss(prediction, label)
            # if args.contras:
            #     loss = (loss + itc_loss).mean()
            accuracy = accuracy.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            step += 1
            if step % args.print_steps == 0 and args.rank == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}, "
                      f"lr {scheduler.get_last_lr()[0]}")

        # 4. validation
        start_time = time.time()
        loss, macro_f1 = validate(model, val_dataloader, device, args)
        end_time = time.time()
        total_time = (end_time - start_time) / 10000

        if args.rank == 0:
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, macro f1: {macro_f1:.3f}, time: {total_time}")
            print(f"Epoch {epoch} step {step}: loss {loss:.3f}, macro f1: {macro_f1:.3f}, time: {total_time}")

        # 5. save checkpoint
        if args.rank == 0:
            macro_f1 = round(macro_f1, 4)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'macro_f1': macro_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}_macro_f1_{macro_f1}_{epoch}.bin')
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    if args.distributed:
        init_distributed_mode(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
