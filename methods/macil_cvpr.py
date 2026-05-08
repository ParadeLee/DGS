import torch
import time
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.network_newlora import DGSNet
from models.attention import Attention_newLoRA

from utils.schedulers import cosine_lr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import count_parameters, compute_flops
from models.losses import AngularPenaltySMLoss, compute_angle_weighted_patch_distillation_loss
import re
import optimgrad
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = DGSNet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_newLoRA):
                module.init_param()

        self.args = args
        # self.writer = SummaryWriter(log_dir=os.path.join(self.args['tblogdir'], f'_{self.args["timestamp"]}'))
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]

        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.fc_lrate = args["fc_lrate"] 
        self.thres = 0.99 # svd截断分数
        self.tau = args["tau"]

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False
        self.fea_proj = defaultdict(dict)
        self.logit_norm = 0.1  
        
        self.task_sizes = []
        self.inc_classes = 0
        
        self._class_means = None
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self._old_class_covs = None
        self._compute_FLOPS(self._network)
    
    def _compute_FLOPS(self, model):
        model.eval()
        flops = compute_flops(model, input_shape=(1, 3, 224, 224), device=self._device)
        logging.info('FLOPS: {:.2f}G'.format(flops/1e9))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.inc_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.inc_classes
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)
        
        # Semantic Shift old embedding 
        time_start = time.time()
        self._train(self.train_loader, self.test_loader)
        time_end = time.time()
        logging.info('Backbone Training time: {:.2f}s'.format(time_end - time_start))
        self.fea_proj = defaultdict(dict)

        # Semantic Shift
        if self._cur_task > 0:
            time_start = time.time()
            self._old_network.to(self._device)
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._old_network)
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 1.0)
            old_class_mean += gap
            self._class_means[:self._known_classes] = old_class_mean

        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

        if self._cur_task > 0:
            self._stage2_compact_classifier(self.task_sizes[-1])
            time_end = time.time()
            logging.info('CA Training time: {:.2f}s'.format(time_end - time_start))
            pass
    
    def svd_fea(self, matrix, thres):
        _, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        V = Vh.T
        cumulative = S.cumsum(0) / S.sum()
        num_vec = (cumulative >= thres).nonzero(as_tuple=True)[0][0] + 1
        basis = V[:, :num_vec]
        proj = basis @ basis.T
        proj = proj / torch.norm(proj)
        return proj


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        network_params = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if re.search(rf"(^|\.)classifier_pool\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})

            if re.search(rf"(^|\.)lora_B_q\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_A_q\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
            if re.search(rf"(^|\.)lora_A_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                param.requires_grad_(True)
                network_params.append({'params': param})
                
        with torch.no_grad():
            if self._cur_task > 0:
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    self._network(inputs, get_cur_feat=True)
                for module in self._network.modules():
                    if isinstance(module, Attention_newLoRA):
                        proj = self.svd_fea(module.cur_matrix, self.thres)
                        self.fea_proj[module.lora_A_q[self._cur_task].weight] = proj.to(self._device)
                        self.fea_proj[module.lora_A_v[self._cur_task].weight] = proj.to(self._device)
                        self.fea_proj[module.lora_B_q[self._cur_task].weight] = proj.to(self._device)
                        self.fea_proj[module.lora_B_v[self._cur_task].weight] = proj.to(self._device)
                        module.cur_matrix.zero_() 
                        module.n_cur_matrix = 0
            if self._cur_task == 0:
                self.run_epoch = self.init_epoch
            else:
                self.run_epoch = self.epochs
        train_iter = iter(train_loader)
        num_batches = len(train_loader)
        total_iterations = self.run_epoch * num_batches
        model_optimizer = torch.optim.SGD(network_params, lr=self.init_lr, weight_decay=self.weight_decay,  momentum=0.9)  # 初始化优化器
        model_scheduler = cosine_lr(model_optimizer, self.init_lr, 30, total_iterations)

        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        # logging.info(f"Parameters to be updated: {enabled}")

        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])

        self._network.train()
        losses = 0.
        correct, total = 0, 0
        for iteration in tqdm(range(total_iterations + 1)):
            model_scheduler(iteration)
            try:
                _, inputs, targets = next(train_iter)
            except:
                train_iter = iter(train_loader)
                _, inputs, targets = next(train_iter)
            
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask) - self._known_classes
            output = self._network(inputs)
            logits = output['logits']
            features = output['features']
            patch_tokens = output['patch_tokens']

            ce_loss = loss_cos(logits, targets) 
            patch_loss = 0.0
            if self._cur_task > 0:
                with torch.no_grad():
                    old_output = self._old_network(inputs) 
                    old_patch_tokens = old_output['patch_tokens'] 
                patch_loss = compute_angle_weighted_patch_distillation_loss(patch_tokens, old_patch_tokens, features) * self.args["lambada"]
            loss = ce_loss + patch_loss

            model_optimizer.zero_grad()
            loss.backward()
            # 修改gradient
            if self._cur_task > 0:
                for name, param in self._network.named_parameters():
                    if param.requires_grad and param in self.fea_proj:
                        proj = self.fea_proj[param]
                        proj.detach_()
                        grad = param.grad.data
                        if proj.shape[0] == grad.shape[0]:
                            grad_p = torch.mm(proj, grad)
                        else:
                            grad_p = torch.mm(grad, proj)
                        param.grad.data = self.tau * grad_p + (1 - self.tau) * grad

            model_optimizer.step()

            losses += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        # task train finished
        test_acc = self._compute_accuracy(self._network, test_loader)
        final_info = 'Task {}, Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, 
                losses / (len(train_loader)*self.run_epoch), 
                train_acc,
                test_acc,
            )
        logging.info(final_info)


    
    def accuracy(self, y_pred, y_true, accuracy_matrix = False):
        assert len(y_pred) == len(y_true), 'Data length error.'
        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
        i = 0
        # Grouped accuracy
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+self.class_num-1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label] 
            i += 1
        # Old accuracy
        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
        # New accuracy
        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        # print(len(y_pred), len(y_true))
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                task_id = (targets//self.class_num).cpu()
                y_true_task.append(task_id)
                outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def extract_features(self, trainloader, model, task_id = None):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data, task_id)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):

        param_list = []
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)

            if re.search(rf"(^|\.)classifier_pool\.(\d+)($|\.)", name) is not None:
                idx = int(re.search(rf"(^|\.)classifier_pool\.(\d+)($|\.)", name).group(2))
                if idx < self._network.numtask:
                    param.requires_grad_(True)
                    param_list.append( param )
            
        run_epochs = ca_epochs
        crct_num = self._total_classes    
        network_params = [{'params': param_list, 'lr': self.fc_lrate, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.fc_lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        
        for epoch in range(run_epochs):
            self._network.to(self._device)
            self._network.eval()
            losses = 0.
            ce_losses = 0.
            reg_losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256 # origin
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = self._class_means[c_id].to(self._device)*(0.9+decay)
                cls_cov = self._class_covs[c_id].to(self._device)  
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            # 修改ca的loss
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls] 

                # -stage two only use classifiers
                logits = self._network(inp, fc_only=True)

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True) 
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True)
                    ce_loss = F.cross_entropy(decoupled_logits, tgt)
                    # 这里未进行norm设计，无法调用ce_loss
                    # ce_loss = loss_cos(norms_all, tgt[:crct_num])
                loss = ce_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_losses += ce_loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} =>celoss:{:.3f}, Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, ce_losses/self._total_classes, losses/self._total_classes, test_acc)
            logging.info(info)
 
 
    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        # covariance
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = torch.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov

        elif not check_diff:
            self._class_means = torch.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors_ts = torch.tensor(vectors, dtype=torch.float32)
            class_mean = torch.mean(torch.tensor(vectors), dim=0)
            class_cov = torch.cov(vectors_ts.T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean.detach()
            self._class_covs[class_idx, ...] = class_cov.detach()
   
    def displacement(self, Y1, Y2, embedding_old, sigma=1.41):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
    

    def shrink_cov(self, cov):
        alpha1 = 10
        alpha2 = 10
        # Compute the mean of the diagonal elements
        diag_mean = torch.mean(torch.diagonal(cov))
        # Create a copy of the covariance matrix with zeroed out diagonals
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        # Compute the mean of the off-diagonal elements (non-zero entries)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag * mask).sum() / mask.sum()
        # Identity matrix
        iden = torch.eye(cov.size(0), device=cov.device)
        # Shrink the covariance matrix
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        return cov_
