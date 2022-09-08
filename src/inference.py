import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from config import parse_args
from dataset.data_helper import SampleDataset
from models.modal_classifiser import SampleModel
import numpy as np
import time
import os


def inference(args):
    device = torch.device(args.device)
    # 2. load model
    model = SampleModel(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
        
    # 1. load data
    dataset = SampleDataset(args, args.test_data, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.test_batch_size, sampler=sampler, drop_last=False,
        pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch
    )

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_input, text_mask = batch['text_input'].to(device), batch['text_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            pred_label_id = model(text_input, text_mask, token_type_ids)
            predictions.extend(pred_label_id.cpu().numpy())
    
    predictions = np.array(predictions)

    predictions = np.argmax(predictions, axis=1).tolist()
    # 4. dump results
    with open(args.submission_csv, 'w') as f:
        f.write('id,label\n')
        for pred_label_id, data in zip(predictions, dataset.data):
            id = data['id']
            f.write(f'{id},{pred_label_id}\n')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    inference(args)
    
