import time
import numpy as np
import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from config import get_config
from models import SNGNN, SNGNN_Plus, \
    MLP, LINK, GAT, MixHop, GCNJK, GCNII, GPRGNN, \
    LINKX, H2GCN, APPNP_Net, LINK_Concat, MLPNORM,\
    GGCN, ACMGCN, WRGAT, AGNN
from datasets import SNGNNPlanetoid, SNGNNWebKB, SNGNNActor, SNGNNWikipediaNetwork
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, is_undirected, to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from utils import set_random_seed, mkdir_or_exist, get_root_logger, lexsort_torch
from utils.data_transform import dense_to_sparse_coo_tensor, edge_index_to_adj_mx, \
    row_normalize, sparse_mx_to_torch_sparse_tensor, edge_index_to_torch_coo_tensor, \
    row_normalized_adjacency, get_adj_high, cosine_similarity

def parse_args():
    parser = argparse.ArgumentParser(description='Train a graph neural network')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='config file path')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset to use')
    parser.add_argument('--model', type=str, default='GCN',
                        help='model to choose, for expammle: GCN, SpGAT, GAT')
    parser.add_argument('--work-dir', type=str, default='./work_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train.')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='num of hidden channels for model')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='num of network layers for model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--data-sorting', action='store_true', default=False,
                        help='whether not to sort the data before training')
    parser.add_argument('--data_splits', action='store_true', default=False,
                        help='whether to use the data split in reported paper in Planetoid')
    parser.add_argument('--part_id', type=int, default=0,
                        help='data split part')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='if early_stoppping > 40, using the early_stopping method')
    parser.add_argument('--top_k', type=int, default=1,
                        help='select top_k for V4')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='threshold  for V4')
    parser.add_argument('--init_beta', type=float, default=0.5,
                        help='trade off')
    parser.add_argument('--is_remove_self_loops', type=int, default=1,
                        help='whether to remove self-loops, 1 True, 0 False')

    args = parser.parse_args()

    return args


def train_step(model, data, optimizer, cfg, extra_info):
    model.train()
    optimizer.zero_grad()
    if cfg['model'] in ['ACMGCN', 'GGCN', 'MLPNORM']:
        output = model(data, extra_info)
    else:
        output = model(data)

    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    _, pred = output.max(dim=1)
    correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = correct / int(data.train_mask.sum())
    # acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    train_loss.backward()
    optimizer.step()
    # logger.info('Epoch: {:d}, Loss: {:.4f}'.format(epoch, loss))
    return train_loss, train_acc


def validate_step(model, data, cfg, extra_info):
    model.eval()
    with torch.no_grad():
        if cfg['model'] in ['ACMGCN', 'GGCN', 'MLPNORM']:
            output = model(data, extra_info)
        else:
            output = model(data)
        _, pred = output.max(dim=1)
        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_loss = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        val_acc = correct / int(data.val_mask.sum())
        return val_loss, val_acc


def test_step(model, data, cfg, extra_info):
    model.eval()
    with torch.no_grad():
        if cfg['model'] in ['ACMGCN', 'GGCN', 'MLPNORM', ]:
            output = model(data, extra_info)
        else:
            output = model(data)
        _, pred = output.max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        test_loss = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        test_acc = correct / int(data.test_mask.sum())
        return test_loss, test_acc


def train(model, data, optimizer, logger, cfg, extra_info):
    logger.info('Start training...')
    dur = []
    final_test_acc = 0

    #
    bad_counter = 0
    smallest_val_loss = float('inf')

    patience = cfg['patience']
    valoss_mn = float('inf')
    valacc_mx = 0.0
    curr_step = 0

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss, train_acc = train_step(model, data, optimizer, cfg, extra_info)
        val_loss, val_acc = validate_step(model, data, cfg, extra_info)
        test_loss, test_acc = test_step(model, data, cfg, extra_info)
        dur.append(time.time() - t0)
        logger.info('Epoch: {:d} | Train_loss: {:.4f}, Train_acc:{:.4f}, '
                    'Val_loss: {:.4f}, Val_acc:{:.4f}, Test_loss: {:.4f}, Test_acc:{:.4f}, Time(s): {:.4f}'
                    .format(epoch, train_loss, train_acc, val_loss, val_acc,
                            test_loss, test_acc, sum(dur) / len(dur)))

        # version-1
        # if val_acc > valacc_mx:
        #     valacc_mx = val_acc
        #     final_test_acc = test_acc
        #     curr_step = 0
        if val_loss < smallest_val_loss:
            smallest_val_loss = val_loss
            # torch.save(model.state_dict(), checkpt_file)
            final_test_acc = test_acc
            curr_step = 0
        else:
            curr_step += 1
        if curr_step == patience:
            break

    return final_test_acc


def main():
    args = parse_args()

    cfg = get_config(args.config)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg['work_dir'] = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.data_sorting is not None:
        cfg['data_sorting'] = args.data_sorting
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.patience is not None:
        cfg['patience'] = args.patience
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.weight_decay is not None:
        cfg['weight_decay'] = args.weight_decay
    if args.data_splits is not None:
        cfg['data_splits'] = args.data_splits
    if args.part_id is not None:
        cfg['part_id'] = args.part_id
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.hidden_channels is not None:
        cfg['hidden_channels'] = args.hidden_channels
    if args.early_stopping is not None:
        cfg['early_stopping'] = args.early_stopping
    if args.dropout is not None:
        cfg['dropout'] = args.dropout
    if args.dataset is not None:
        cfg['dataset'] = args.dataset
    if args.model is not None:
        cfg['model'] = args.model
    if args.num_layers is not None:
        cfg['num_layers'] = args.num_layers
    if args.no_cuda is not None:
        cfg['no_cuda'] = args.no_cuda
    if args.top_k is not None:
        cfg['top_k'] = args.top_k
    if args.thr is not None:
        cfg['thr'] = args.thr
    if args.init_beta is not None:
        cfg['init_beta'] = args.init_beta
    if args.is_remove_self_loops is not None:
        cfg['is_remove_self_loops'] = args.is_remove_self_loops

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg['work_dir']))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(
        cfg['work_dir'],
        '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(cfg['model'], cfg['dataset'], cfg['lr'], cfg['weight_decay'],
                                                      cfg['dropout'], cfg['hidden_channels'], cfg['num_layers'],
                                                      cfg['top_k'], cfg['thr'], cfg['is_remove_self_loops'],
                                                      cfg['init_beta'], cfg['patience'],
                                                      cfg['part_id']))
    # logger = get_root_logger(log_file=log_file, log_level=cfg['log_level'])
    logger = get_root_logger(cfg['model'], log_file=log_file)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(cfg['seed']))
        set_random_seed(args.seed)
        cfg['seed'] = args.seed
    else:
        set_random_seed(cfg['seed'])

    # logger basic setting
    logger.info(f'Config:\n{cfg}')

    # device
    # device = torch.device('cpu')
    if cfg['no_cuda']:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset setting
    if cfg['dataset'] == 'Cora':
        dataset = SNGNNPlanetoid(
            root='./datasets/data/Cora', name='Cora')
    elif cfg['dataset'] == 'CiteSeer':
        dataset = SNGNNPlanetoid(
            root='./datasets/data/CiteSeer', name='CiteSeer')
    elif cfg['dataset'] == 'PubMed':
        dataset = SNGNNPlanetoid(
            root='./datasets/data/PubMed', name='PubMed')
    elif cfg['dataset'] == 'Cornell':
        dataset = SNGNNWebKB(
            root='./datasets/data/Cornell', name='Cornell')
        dataset.process()
    elif cfg['dataset'] == 'Texas':
        dataset = SNGNNWebKB(
            root='./datasets/data/Texas', name='Texas')
        dataset.process()
    elif cfg['dataset'] == 'Wisconsin':
        dataset = SNGNNWebKB(
            root='./datasets/data/Wisconsin', name='Wisconsin')
        dataset.process()
    elif cfg['dataset'] == 'Actor':
        dataset = SNGNNActor(
            root='./datasets/data/Actor')
        dataset.process()
    elif cfg['dataset'] == 'Chameleon':
        dataset = SNGNNWikipediaNetwork(
            root='./datasets/data/chameleon', name='chameleon')
        dataset.process()
    elif cfg['dataset'] == 'Squirrel':
        dataset = SNGNNWikipediaNetwork(
            root='./datasets/data/squirrel', name='squirrel')
        dataset.process()
    else:
        logger.error('wrong dataset settings')
        return

    extra_info = {}
    if cfg['model'] in ['MLPNORM']:
        adj_dense = to_dense_adj(dataset[0].edge_index)
        extra_info.setdefault('adj_dense', adj_dense.to(device))
    if cfg['model'] in ['GGCN']:
        adj_coo_tensor = edge_index_to_torch_coo_tensor(dataset[0].x, dataset[0].edge_index).to(device)
        extra_info.setdefault('adj_coo_tensor', adj_coo_tensor)
    if cfg['model'] == 'ACMGCN':
        adj_low = to_scipy_sparse_matrix(dataset[0].edge_index)
        adj_low = row_normalized_adjacency(adj_low)
        adj_high = get_adj_high(adj_low)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high)
        extra_info.setdefault('adj_low', adj_low.to(device))
        extra_info.setdefault('adj_high', adj_high.to(device))


    # model setting
    if cfg['model'] == 'GCN':
        model = GCN(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'])
    elif cfg['model'] == 'GAT':
        # model = GATNet(dataset.num_features, dataset.num_classes)
        model = GAT(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'])
    elif cfg['model'] == 'SNGNN':
        # edge_index, _ = add_self_loops(dataset[0].edge_index, num_nodes=dataset[0].x.size(0))
        model = SNGNN(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'])
    elif cfg['model'] == 'SNGNN_Plus':
        model = SNGNN_Plus(dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
                             dataset[0].num_nodes, cfg['num_layers'], cfg['top_k'], cfg['thr'],
                             cfg['is_remove_self_loops'], cfg['dropout'])
    elif cfg['model'] == 'SNGNN_Plus_Plus':
        model = SNGNN_Plus_Plus(dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
                             dataset[0].num_nodes, cfg['num_layers'], cfg['top_k'], cfg['thr'], cfg['init_beta'],
                             cfg['is_remove_self_loops'], cfg['dropout'])
        # model = SNGNN_Plus(dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
        #                      dataset[0].num_nodes, cfg['num_layers'], cfg['top_k'], cfg['thr'],
        #                      cfg['is_remove_self_loops'], cfg['dropout'])
    elif cfg['model'] == 'AGNN':
        model = AGNN(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'])
    elif cfg['model'] == 'MLP':
        model = MLP(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'])
    elif cfg['model'] == 'LINK':
        model = LINK(dataset[0].num_nodes, dataset.num_classes)
    elif cfg['model'] == 'MixHop':
        model = MixHop(dataset.num_features, cfg['hidden_channels'],
                       dataset.num_classes, num_layers=2, dropout=0.5, hops=2)
    elif cfg['model'] == 'GCNJK':
        model = GCNJK(dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
                      num_layers=2, dropout=0.5, save_mem=False, jk_type='max')
    elif cfg['model'] == 'GATJK':
        model = GCNJK(dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
                      num_layers=2, dropout=0.5, save_mem=False, jk_type='max')
    elif cfg['model'] == 'GCNII':
        # --alpha 0.0 --beta 1.0 --gamma 0.7 --delta 0.1
        model = GCNII(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'], 0.0, 1.0)
    elif cfg['model'] == 'GPRGNN':
        model = GPRGNN(dataset.num_features, cfg['hidden_channels'], dataset.num_classes)
    elif cfg['model'] == 'LINKX':
        model = LINKX(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'], dataset[0].num_nodes)
    elif cfg['model'] == 'H2GCN':
        model = H2GCN(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, dataset[0].edge_index, dataset[0].num_nodes)
    elif cfg['model'] == 'APPNP_Net':
        model = APPNP_Net(dataset.num_features, cfg['hidden_channels'], dataset.num_classes)
    elif cfg['model'] == 'LINK_Concat':
        model = LINK_Concat(dataset.num_features, cfg['hidden_channels'], dataset.num_classes, cfg['num_layers'], dataset[0].num_nodes)
    elif cfg['model'] == 'MLPNORM':
        # model = MLPNORM(dataset[0].num_nodes, dataset.num_features, cfg['hidden_channels'], dataset.num_classes,
        #                 dropout, alpha, beta, gamma, delta, norm_func_id, norm_layers, orders, orders_func_id, device)
        model = MLPNORM(dataset[0].num_nodes, dataset.num_features, 256, dataset.num_classes, 0.5, 0, 1, 0.5, 0.5,
                 1, 2, 2, 2, device)
    elif cfg['model'] == 'GGCN':
        # model = GGCN(nfeat=d, nlayers=args.num_layers, nhidden=args.hidden_channels, nclass=c, dropout=args.dropout,
        #              decay_rate=args.decay_rate, exponent=args.exponent, device=device, use_degree=False,
        #              use_sign=True, use_decay=True, use_sparse=True, scale_init=0.5, deg_intercept_init=0.5,
        #              use_bn=False, use_ln=False).to(device)
        model = GGCN(nfeat=dataset.num_features, nlayers=cfg['num_layers'], nhidden=cfg['hidden_channels'],
                     nclass=dataset.num_classes, dropout=0.0, decay_rate=1e-7, exponent=2, device=device,
                     use_degree=False, use_sign=True, use_decay=True, use_sparse=True, scale_init=0.5,
                     deg_intercept_init=0.5, use_bn=False, use_ln=False)
    elif cfg['model'] == 'ACMGCN':
        # ACMGCN(nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout,
        #        model_type='acmgcn', nlayers=args.num_layers, variant=False).to(device)
        model = ACMGCN(nfeat=dataset.num_features, nhid=cfg['hidden_channels'], nclass=dataset.num_classes,
                       dropout=0, model_type='acmgcn', nlayers=args.num_layers, variant=False)
    # elif cfg['model'] == 'WRGAT':
    #     # model = WRGAT(num_features=d, num_classes=c, num_relations=num_relations,
    #     #               dims=args.hidden_channels, drop=args.dropout).to(device)
    #     model = WRGAT(num_features=dataset.num_features, num_classes=dataset.num_classes, num_relations=num_relations,
    #                   dims=cfg['hidden_channels'], drop=0.0).to(device)
    else:
        logger.error('wrong model settings')
        return

    # logger.info('model:', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    logger.info('number of epoch: {}'.format(cfg['epochs']))

    # check wherther is undirected graph
    # if not is_undirected(dataset[0].edge_index):
    #     logger.info('{} is not undirected'.format(cfg['dataset']))
    #     edge_index = to_undirected(dataset[0].edge_index)
    #     dataset[0].edge_index = edge_index
    #
    # is_normalize = True
    # if is_normalize:
    #     dataset[0].x = F.normalize(dataset[0].x, p=1, dim=-1)

    # begin training
    if cfg['data_sorting']:
        data = dataset[0].to(torch.device('cpu'))
        features, edge_index, labels = lexsort_torch(data.x, data.edge_index, data.y)
        data.x, data.edge_index, data.y = features, edge_index, labels
        # data = Data(x=features, edge_index=edge_index.t().contiguous(), y=labels).to(device)
        data = data.to(device)
    else:
        data = dataset[0].to(device)

    splits_dataset = ['Cornell', 'Texas', 'Wisconsin', 'Actor', 'Chameleon', 'Squirrel']
    if cfg['data_splits']:
        splits_dataset += ['Cora', 'CiteSeer', 'PubMed']
    if cfg['dataset'] in splits_dataset:
        logger.info('Start training dataset {} part {}'.format(cfg['dataset'], cfg['part_id']))
        tmp_train_mask = data.train_mask
        tmp_val_mask = data.train_mask
        tmp_test_mask = data.train_mask
        data.train_mask = data.train_mask[cfg['part_id']]
        data.val_mask = data.val_mask[cfg['part_id']]
        data.test_mask = data.test_mask[cfg['part_id']]
        train_mask_len = len(torch.where(data.train_mask == 1)[0])
        val_mask_len = len(torch.where(data.val_mask == 1)[0])
        test_mask_len = len(torch.where(data.test_mask == 1)[0])
        logger.info('train dataset len:{}, val dataset len:{}, test dataset len:{}'.format(train_mask_len,
                                                                                           val_mask_len,
                                                                                           test_mask_len))
        final_test_acc = train(model.to(device), data, optimizer, logger, cfg, extra_info)
        data.train_mask = tmp_train_mask
        data.val_mask = tmp_val_mask
        data.test_mask = tmp_test_mask
        logger.info('Part {} final test acc: {:.4f}'.format(cfg['part_id'], final_test_acc))
    else:
        final_test_acc = train(model.to(device), data, optimizer, logger, cfg, extra_info)
        logger.info('Part {} final test acc: {:.4f}'.format(cfg['part_id'], final_test_acc))


if __name__ == '__main__':
    main()
