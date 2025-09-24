import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_conv import Network
from code.utils.data_val import get_train_loader, get_test_loader, PolypObjDataset
from code.utils.utils import clip_gradient, adjust_lr, get_coef, cal_ual, adjust_lr_new
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
import tqdm
import random
import argparse
import shutil
import cv2
from torch.utils.data import DataLoader
from code.SAM.DPC import process_selected_samples
#from SAM2_p_b_N import process_selected_samples



os.chdir(os.path.dirname(os.path.abspath(__file__)))

class TrainingState:
    def __init__(self):
        self.last_filter_epoch = -20
        self.filter_count = 0
        self.last_extension_done = False
        self.last_filter = False

training_state = TrainingState()

def cal_iou(pred, mask):
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = (inter + 1) / (union - inter + 1)
    return wiou.mean()

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(2024)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)
    
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print(f'Loaded model from {opt.load}')

    optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
    save_path = opt.save_path
    
    logging.basicConfig(filename=save_path + 'log.log',
                      format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                      level=logging.INFO, 
                      filemode='a', 
                      datefmt='%Y-%m-%d %I:%M:%S %p')

    writer = SummaryWriter(save_path + 'summary')
    step = 0

    global training_state
    training_state = TrainingState()

    for epoch in range(1, opt.epoch + 1):
        train_dataset = PolypObjDataset(image_root=opt.train_root + 'image/', 
                                      gt_root=opt.train_root + 'mask/', 
                                      trainsize=opt.trainsize, 
                                      istraining=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                 batch_size=opt.batchsize,
                                                 num_workers=6,
                                                 shuffle=True,
                                                 drop_last=True)
        total_step = len(train_loader)
        
        cur_lr = adjust_lr_new(epoch, opt.top_epoch, opt.epoch, opt.init_lr, opt.top_lr, opt.min_lr, optimizer)
        #cur_lr = adjust_lr(epoch, opt.top_epoch, opt.epoch, opt.init_lr, opt.top_lr, opt.min_lr, optimizer)
        logging.info(f'learning_rate: {cur_lr}')
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        
        model.train()
        loss_all = 0
        epoch_step = 0
        lr = optimizer.param_groups[0]['lr']

        for i, (images, gts, edges) in enumerate(train_loader, start=1):  
            optimizer.zero_grad()
            images = images.to(device)
            gts = gts.to(device)
            edges = edges.to(device)
            preds = model(images)  

            ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
            ual_loss *= ual_coef

            loss_init = structure_loss(preds[0], gts)*0.0625 + structure_loss(preds[1], gts)*0.125 + structure_loss(preds[2], gts)*0.25 + structure_loss(preds[3], gts)*0.5
            loss_final = structure_loss(preds[4], gts)
            loss_edge = dice_loss(preds[6], edges)*0.125 + dice_loss(preds[7], edges)*0.25 + dice_loss(preds[8], edges)*0.5

            loss = loss_init + loss_final + loss_edge * 4 + 2 * ual_loss
            loss.backward()
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], LR {lr:.8f} Total_loss: {loss.item():.4f} Loss1: {loss_init.item():.4f} Loss2: {loss_final.item():.4f} Loss3: {loss_edge.item():.4f}')
                logging.info(f'[Train Info]:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], Total_loss: {loss.item():.4f} Loss1: {loss_init.item():.4f} Loss2: {loss_final.item():.4f} Loss3: {loss_edge.item():.4f}')
                writer.add_scalars('Loss_Statistics', 
                                 {'Loss_init': loss_init.item(),
                                  'Loss_final': loss_final.item(),
                                  'Loss_edge': loss_edge.item(),
                                  'Loss_total': loss.item()}, 
                                 global_step=step)

        loss_all /= epoch_step
        logging.info(f'[Train Info]: Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_AVG: {loss_all:.4f}')
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if epoch > 99 and epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + f'Net_epoch_{epoch}.pth')

        if epoch >= 150 and (epoch-150) % 20 == 0:
            val_loader = get_test_loader(image_root=opt.val_root + 'image/',
                                       gt_root=opt.val_root + 'mask/',
                                       trainsize=opt.trainsize,
                                       num_workers=12,
                                       batchsize=16)
            val(val_loader, model, epoch, save_path, writer, opt)
        
        if epoch >= 150 and (epoch-150) % 10 == 0:
            test_loader = get_test_loader(
                image_root=opt.test_root + 'image/',
                gt_root=opt.test_root + 'mask/',
                trainsize=opt.trainsize,
                num_workers=12,
                batchsize=16
            )
            val_test(test_loader, model, epoch, save_path, writer, opt)

    writer.close()

def val(test_loader, model, epoch, save_path, writer, opt):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        train_image_dir = os.path.join(opt.train_root, 'image')
        existing_files = set([f for f in os.listdir(train_image_dir) if f.endswith('.jpg')])
        
        filtered_indices = []
        for idx in range(len(test_loader.dataset)):
            image_path = test_loader.dataset.images[idx]
            fname = os.path.basename(image_path)
            if fname not in existing_files:
                filtered_indices.append(idx)
        
        from torch.utils.data import Subset
        filtered_dataset = Subset(test_loader.dataset, filtered_indices)
        filtered_test_loader = DataLoader(
            filtered_dataset, 
            batch_size=opt.batchsize, 
            shuffle=False, 
            num_workers=test_loader.num_workers
        )

        if not training_state.last_extension_done:
            mae_sum = []
            for i, batch in tqdm.tqdm(enumerate(filtered_test_loader, start=1)):
                if len(batch) == 4:
                    image, gt, size_info, name = batch  
                    H, W = size_info
                else:
                    image, gt, edge = batch 
                    H, W = gt.shape[-2], gt.shape[-1]
                    name = [f"batch_{i}_idx_{j}" for j in range(gt.shape[0])]
                
                image = image.to(device)
                gt = gt.to(device)
                
                results = model(image)  
                res = results[4].sigmoid()
                
                for idx in range(len(res)):
                    pre = F.interpolate(res[idx].unsqueeze(0), size=(W[idx].item(), H[idx].item()), mode='bilinear')
                    pre_np = (pre.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    edge = results[8][idx].sigmoid()
                    edge = F.interpolate(edge.unsqueeze(0), size=(W[idx].item(), H[idx].item()), mode='bilinear')
                    edge_np = (edge.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    val_dir = '../val_result'
                    os.makedirs(os.path.join(val_dir, 'mask'), exist_ok=True)
                    os.makedirs(os.path.join(val_dir, 'edge'), exist_ok=True)
                    base_name, _ = os.path.splitext(name[idx])
                    filename = f"{base_name}.png"
                    cv2.imwrite(os.path.join(val_dir, 'mask', filename), pre_np)                   
                    cv2.imwrite(os.path.join(val_dir, 'edge', filename), edge_np)

            if training_state.last_filter and training_state.last_extension_done == False:
                training_state.last_extension_done = True
                val_dir = '../val_result'
                remaining_files = [f for f in os.listdir(val_dir + '/mask') if f.endswith('.png')]
                
                for rname in remaining_files:
                    name_j = rname.replace('.png', '.jpg')
                    src_img = os.path.join(opt.val_root, 'image', name_j)
                    dest_img = os.path.join(opt.train_root, 'image', name_j)
                    shutil.copy2(src_img, dest_img)
                    
                    shutil.copy2(os.path.join('../val_result/mask', rname), 
                            os.path.join(opt.train_root, 'mask', rname))
                    shutil.copy2(os.path.join('../val_result/edge', rname), 
                            os.path.join(opt.train_root, 'edge', rname))

        if epoch >= 150 and not training_state.last_filter:
            x = len(existing_files)

            from code.EDF import process_folder
            output_dir_b = './temp_b'
            output_dir_c = './temp_c'
            
            entropies = process_folder(input_dir='../val_result/mask', output_dir_b=output_dir_b, output_dir_c=output_dir_c)
            
            sorted_entropies = sorted(entropies, key=lambda x: x[1])
            selected_files = [item[0] for item in sorted_entropies[:x]]
            actual_selected = min(len(sorted_entropies), x)

            sam_image_dir = os.path.join(opt.val_root, 'image')  
            sam_mask_dir = output_dir_c  
            output_fuse_dir = './fuse-mask'  
            output_sam_dir = './sam-mask' 

            os.makedirs(output_fuse_dir, exist_ok=True)
            os.makedirs(output_sam_dir, exist_ok=True)

            process_selected_samples(
                image_dir=sam_image_dir,
                mask_dir=sam_mask_dir,
                output_dir=output_fuse_dir,
                selected_files=selected_files  
            )

            for fname in selected_files:
                jpg_fname = fname.replace('.png', '.jpg')
                src_img = os.path.join(opt.val_root, 'image', jpg_fname)
                dest_img = os.path.join(opt.train_root, 'image', jpg_fname)
                shutil.copy2(src_img, dest_img)
                
                fuse_mask_path = os.path.join(output_fuse_dir, fname)
                shutil.copy2(fuse_mask_path, os.path.join(opt.train_root, 'mask', fname))
                shutil.copy2(os.path.join('../val_result/edge', fname), 
                           os.path.join(opt.train_root, 'edge', fname))

            # shutil.rmtree(output_dir_b, ignore_errors=True)
            # shutil.rmtree(output_dir_c, ignore_errors=True)
            if actual_selected < x:
                print(f"第{epoch}轮扩展：合格样本不足，仅扩展{actual_selected}个")
                training_state.last_filter = True

        val_result_dir = '../val_result'
        for subdir in ['mask', 'edge']:
            dir_path = os.path.join(val_result_dir, subdir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)

        if training_state.last_extension_done:
            shutil.rmtree('../val_result', ignore_errors=True)
            return
    torch.cuda.empty_cache()

def val_test(test_loader, model, epoch, save_path, writer, opt):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mae_sum = []
        iou_sum = []
        for i, (image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
            gt = gt.cuda()
            image = image.cuda()
            results = model(image)
            res = results[4]
            res = res.sigmoid()
            iou_sum.append(cal_iou(res, gt).item())
            for i in range(len(res)):
                pre = F.interpolate(res[i].unsqueeze(0), size=(H[i].item(), W[i].item()), mode='bilinear')
                gt_single = F.interpolate(gt[i].unsqueeze(0), size=(H[i].item(), W[i].item()), mode='bilinear')
                mae_sum.append(torch.mean(torch.abs(gt_single - pre)).item())
        iou = np.mean(iou_sum)
        mae = np.mean(mae_sum)
        mae = "%.5f" % mae
        mae = float(mae)
        print(f'Epoch: {epoch}, MAE: {mae}, IoU: {iou}, bestMAE: {opt.best_mae}, bestEpoch: {opt.best_epoch}.')
        if mae < opt.best_mae:
            opt.best_mae = mae
            opt.best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
            print(f'Save state_dict successfully! Best epoch: {epoch}.')
        logging.info(f'[Val Info]:Epoch: {epoch}, MAE: {mae}, IoU: {iou}, bestMAE: {opt.best_mae}, bestEpoch: {opt.best_epoch}.')

    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--top_epoch', type=int, default=10)
    parser.add_argument('--top_lr', type=float, default=1e-4)
    parser.add_argument('--init_lr', type=float, default=1e-7)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--ration', type=str, default='1')
    parser.add_argument('--train_root', type=str, default='../data/TrainDataset/CAMO_COD_train_')
    parser.add_argument('--val_root', type=str, default='../data/TrainDataset/CAMO_COD_generate_')
    parser.add_argument('--best_mae', type=float, default=1.0)
    parser.add_argument('--best_epoch', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='../weight/NNet/')
    parser.add_argument('--test_root', type=str, default='../data/CAMO_TestingDataset/')
    opt = parser.parse_args()

    opt.train_root = opt.train_root + opt.ration + '%/'
    opt.val_root = opt.val_root + str(100-int(opt.ration)) + '%/'
    opt.save_path = opt.save_path + opt.ration + "%/"

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    train(opt)

if __name__ == '__main__':
    main()