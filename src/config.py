import argparse
import os

base_dir = os.path.dirname(__file__)


def parse_args():
    data_path = os.path.join(base_dir, '../data/')
    root_path = os.path.join(base_dir, '../')
    
    parser = argparse.ArgumentParser(description="CCF BDCI 小样本数据分类任务")
    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= train config ===========================
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.5)

    # ========================= Data Configs ==========================
    parser.add_argument('--train_data', type=str, default=data_path + 'train.json')
    parser.add_argument('--test_data', type=str, default=data_path + 'testA.json')
    parser.add_argument('--submission_csv', type=str, default=root_path + 'results/result.csv')
    
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=16, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=16, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=0, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='data/save/v1')
    parser.add_argument('--ckpt_file', type=str, default='data/save/v1/model_.bin')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=6, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=500, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    

    # ========================== BERT =============================
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--bert_dir', type=str, default=root_path + 'opensource_models/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=320)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # DDP
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cpu')
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument('--rank', type=int, default=0)

    return parser.parse_args()
