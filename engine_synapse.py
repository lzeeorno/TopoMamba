import numpy as np
from tqdm import tqdm
import gc

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume, test_single_volume_dice_only

import time


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "gradient_clip_norm", 12.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "gradient_clip_norm", 12.0))
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0 and iter != 0:
            topo = getattr(criterion, "last_components", {})
            topo_txt = f", base: {topo.get('base', 0):.4f}, topo: {topo.get('topology', 0):.4f}" if topo else ""
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}{topo_txt}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    return mean_loss


def calculate_dice_coefficient(pred, target, num_classes):
    """计算每个类别的Dice系数"""
    dice_scores = []
    
    for class_idx in range(1, num_classes):  # 跳过背景类
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union == 0:
            dice = 1.0  # 如果预测和真实都为0，认为完全匹配
        else:
            dice = (2.0 * intersection) / union
        
        # 如果dice是tensor，则调用.item()，否则直接添加
        if hasattr(dice, 'item'):
            dice_scores.append(dice.item())
        else:
            dice_scores.append(dice)
    
    return dice_scores


def train_one_epoch_with_metrics(train_loader,
                                model,
                                criterion, 
                                optimizer, 
                                scheduler,
                                epoch, 
                                logger, 
                                config, 
                                scaler=None):
    '''
    train model for one epoch with detailed metrics
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []
    dice_scores_list = []

    accum_steps = getattr(config, 'accum_steps', 1)
    for iter, data in enumerate(train_loader):
        if iter % accum_steps == 0:
            optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss / accum_steps).backward()
            if (iter + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "gradient_clip_norm", 12.0))
                scaler.step(optimizer)
                scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            (loss / accum_steps).backward()
            if (iter + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "gradient_clip_norm", 12.0))
                optimizer.step()

        loss_list.append(loss.item())
        
        # 优化：减少Dice计算频率，只在每10个iteration计算一次，避免频繁的GPU-CPU同步
        if iter % 10 == 0:
            with torch.no_grad():
                pred = torch.argmax(out, dim=1)
                target_int = targets.argmax(dim=1) if targets.dim() > 3 else targets
                dice_scores = calculate_dice_coefficient(pred, target_int, config.num_classes)
                dice_scores_list.append(dice_scores)
        
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0 and iter != 0:
            topo = getattr(criterion, "last_components", {})
            topo_txt = f", base: {topo.get('base', 0):.4f}, topo: {topo.get('topology', 0):.4f}" if topo else ""
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}{topo_txt}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    
    scheduler.step()
    
    # 计算平均Dice系数
    if dice_scores_list:
        dice_scores_array = np.array(dice_scores_list)
        avg_dice_per_organ = np.mean(dice_scores_array, axis=0)
        overall_avg_dice = np.mean(avg_dice_per_organ)
    else:
        avg_dice_per_organ = [0.0] * (config.num_classes - 1)
        overall_avg_dice = 0.0
    
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, avg_dice: {overall_avg_dice:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    
    return mean_loss, avg_dice_per_organ, overall_avg_dice


def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test, 
                                    network_type=config.network, postprocess_config=config)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95


def val_one_epoch_fast_dice_only(test_datasets,
                                test_loader,
                                model,
                                epoch, 
                                logger,
                                config,
                                test_save_path,
                                val_or_test=False):
    """
    快速验证函数，只计算Dice指标，跳过HD95计算以提高速度
    专门用于训练期间的中间验证，最后一个epoch再使用完整的验证函数
    """
    stime = time.time()
    model.eval()
    with torch.no_grad():
        dice_list = []
        i_batch = 0
        for data in tqdm(test_loader, desc=f"Fast Val Epoch {epoch}"):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            # 只计算Dice，跳过HD95
            dice_i = test_single_volume_dice_only(img, msk, model, classes=config.num_classes, 
                                                patch_size=[config.input_size_h, config.input_size_w],
                                                test_save_path=test_save_path, case=case_name, 
                                                network_type=config.network)
            dice_list.append(dice_i)

            logger.info('idx %d case %s mean_dice %f (HD95 skipped)' % (i_batch, case_name, np.mean(dice_i)))
            i_batch += 1
            
        # 计算平均Dice
        dice_array = np.array(dice_list)
        mean_dice = np.mean(dice_array)
        
        # 记录每个器官的Dice
        for i in range(1, config.num_classes):
            organ_dice = np.mean(dice_array[:, i-1])
            logger.info('Mean class %d mean_dice %f (HD95 skipped)' % (i, organ_dice))
        
        etime = time.time()
        log_info = f'fast val epoch: {epoch}, mean_dice: {mean_dice}, time(s): {etime-stime:.2f} (HD95 skipped for speed)'
        print(log_info)
        logger.info(log_info)
    
    return mean_dice


def val_one_epoch_with_detailed_metrics(test_datasets,
                                       test_loader,
                                       model,
                                       epoch, 
                                       logger,
                                       config,
                                       test_save_path,
                                       val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test,
                                    network_type=config.network, postprocess_config=config)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        
        # 提取每个器官的dice和hd95
        dice_per_organ = []
        hd95_per_organ = []
        
        for i in range(1, config.num_classes):
            dice_score = metric_list[i-1][0]
            hd95_score = metric_list[i-1][1]
            dice_per_organ.append(dice_score)
            hd95_per_organ.append(hd95_score)
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, dice_score, hd95_score))
        
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95, dice_per_organ, hd95_per_organ


def val_one_epoch_with_visualization(test_datasets,
                                   test_loader,
                                   model,
                                   epoch, 
                                   logger,
                                   config,
                                   test_save_path,
                                   prediction_vis_dir=None,
                                   attention_vis_dir=None,
                                   activation_vis_dir=None,
                                   val_or_test=False,
                                   save_vis_every_n=5,
                                   save_prediction_comparison_func=None,
                                   extract_attention_maps_func=None,
                                   save_attention_heatmaps_func=None,
                                   extract_activation_maps_func=None,
                                   save_activation_heatmaps_func=None):
    """
    验证函数，支持生成预测对比图、注意力热图和激活热图
    Args:
        save_vis_every_n: 每n个案例保存一次可视化图像
        save_prediction_comparison_func: 保存预测对比图的函数
        extract_attention_maps_func: 提取注意力图的函数
        save_attention_heatmaps_func: 保存注意力热图的函数
        extract_activation_maps_func: 提取激活图的函数
        save_activation_heatmaps_func: 保存激活热图的函数
        activation_vis_dir: 激活热图保存目录
    """
    
    stime = time.time()
    model.eval()
    
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0

        def get_expected_input_channels():
            expected_in_chans = getattr(config, 'input_channels', None)
            model_config = getattr(config, 'model_config', None)
            if isinstance(model_config, dict):
                expected_in_chans = model_config.get('input_channels', expected_in_chans)

            try:
                module = model.module if hasattr(model, 'module') else model
                patch_proj = getattr(getattr(module, 'patch_embed', None), 'proj', None)
                if patch_proj is not None and hasattr(patch_proj, 'in_channels'):
                    expected_in_chans = int(patch_proj.in_channels)
            except Exception:
                pass

            try:
                return max(1, int(expected_in_chans))
            except (TypeError, ValueError):
                return 1

        expected_in_chans = get_expected_input_channels()

        def build_visualization_input_tensor(slice_img):
            input_tensor = torch.from_numpy(slice_img).float().cuda()
            if expected_in_chans == 1:
                return input_tensor.unsqueeze(0).unsqueeze(0)
            return input_tensor.unsqueeze(0).repeat(expected_in_chans, 1, 1).unsqueeze(0)
        
        # 用于收集每个样本的平均Dice和HD95，以便计算标准差
        all_sample_dice = []
        all_sample_hd95 = []
        
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            # 执行预测并计算指标
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, 
                                        patch_size=[config.input_size_h, config.input_size_w],
                                        test_save_path=test_save_path, case=case_name, 
                                        z_spacing=config.z_spacing, val_or_test=val_or_test,
                                        network_type=config.network, postprocess_config=config)
            metric_list += np.array(metric_i)
            
            # 记录每个样本的平均Dice和HD95
            sample_mean_dice = np.mean(metric_i, axis=0)[0]
            sample_mean_hd95 = np.mean(metric_i, axis=0)[1]
            all_sample_dice.append(sample_mean_dice)
            all_sample_hd95.append(sample_mean_hd95)
            
            # 每隔save_vis_every_n个案例生成可视化图像
            if i_batch % save_vis_every_n == 0 and (prediction_vis_dir or attention_vis_dir or activation_vis_dir):
                try:
                    # 生成预测结果用于可视化
                    img_np = img.squeeze(0).cpu().detach().numpy()
                    msk_np = msk.squeeze(0).cpu().detach().numpy()
                    
                    # 创建模型预测
                    if len(img_np.shape) == 3:  # 3D数据
                        prediction_np = np.zeros_like(msk_np)
                        
                        # 为3D数据生成注意力热图和激活热图 - 完整的3D数据处理
                        if attention_vis_dir and extract_attention_maps_func and save_attention_heatmaps_func:
                            try:
                                mid_slice = img_np[img_np.shape[0] // 2]
                                x, y = mid_slice.shape[0], mid_slice.shape[1]
                                if x != config.input_size_h or y != config.input_size_w:
                                    from scipy.ndimage import zoom
                                    mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                input_tensor = build_visualization_input_tensor(mid_slice)
                                
                                attention_maps = extract_attention_maps_func(model, input_tensor)
                                if attention_maps:
                                    save_attention_heatmaps_func(
                                        img_np, attention_maps, case_name, attention_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成注意力热图失败 {case_name}: {e}")
                        
                        # 为3D数据生成激活热图
                        if activation_vis_dir and extract_activation_maps_func and save_activation_heatmaps_func:
                            try:
                                mid_slice = img_np[img_np.shape[0] // 2]
                                x, y = mid_slice.shape[0], mid_slice.shape[1]
                                if x != config.input_size_h or y != config.input_size_w:
                                    from scipy.ndimage import zoom
                                    mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                input_tensor = build_visualization_input_tensor(mid_slice)
                                
                                activation_maps = extract_activation_maps_func(model, input_tensor)
                                if activation_maps:
                                    save_activation_heatmaps_func(
                                        img_np, activation_maps, case_name, activation_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成激活热图失败 {case_name}: {e}")
                        
                        # 生成每个slice的预测
                        for ind in range(img_np.shape[0]):
                            slice_img = img_np[ind, :, :]
                            x, y = slice_img.shape[0], slice_img.shape[1]
                            
                            # 调整尺寸
                            if x != config.input_size_h or y != config.input_size_w:
                                from scipy.ndimage import zoom
                                slice_img = zoom(slice_img, (config.input_size_h / x, config.input_size_w / y), order=3)
                            
                            input_tensor = build_visualization_input_tensor(slice_img)
                            
                            # 获取预测结果
                            outputs = model(input_tensor)
                            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                            
                            # 调整回原尺寸
                            if x != config.input_size_h or y != config.input_size_w:
                                pred = zoom(pred, (x / config.input_size_h, y / config.input_size_w), order=0)
                            
                            prediction_np[ind] = pred
                    else:
                        # 2D数据处理
                        input_tensor = build_visualization_input_tensor(img_np)
                        
                        outputs = model(input_tensor)
                        prediction_np = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                        
                        # 生成注意力热图
                        if attention_vis_dir and extract_attention_maps_func and save_attention_heatmaps_func:
                            try:
                                attention_maps = extract_attention_maps_func(model, input_tensor)
                                if attention_maps:
                                    save_attention_heatmaps_func(
                                        img_np, attention_maps, case_name, attention_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成注意力热图失败 {case_name}: {e}")
                        
                        # 生成激活热图
                        if activation_vis_dir and extract_activation_maps_func and save_activation_heatmaps_func:
                            try:
                                activation_maps = extract_activation_maps_func(model, input_tensor)
                                if activation_maps:
                                    save_activation_heatmaps_func(
                                        img_np, activation_maps, case_name, 
                                        activation_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成激活热图失败 {case_name}: {e}")
                    
                    # 保存预测对比图
                    if prediction_vis_dir and save_prediction_comparison_func:
                        try:
                            save_prediction_comparison_func(
                                img_np, msk_np, prediction_np, case_name, prediction_vis_dir
                            )
                            logger.info(f"已保存预测对比图: {case_name}")
                        except Exception as e:
                            logger.warning(f"保存预测对比图失败 {case_name}: {e}")
                            
                except Exception as e:
                    logger.warning(f"可视化处理失败 {case_name}: {e}")

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
            
            # 每处理一个样本后清理内存，避免累积
            if i_batch % 2 == 0:  # 每2个样本清理一次
                torch.cuda.empty_cache()
                gc.collect()
        
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        
        # 计算标准差
        std_dice = np.std(all_sample_dice) if len(all_sample_dice) > 0 else 0.0
        std_hd95 = np.std(all_sample_hd95) if len(all_sample_hd95) > 0 else 0.0
        
        # 提取每个器官的dice和hd95
        dice_per_organ = []
        hd95_per_organ = []
        
        for i in range(1, config.num_classes):
            dice_score = metric_list[i-1][0]
            hd95_score = metric_list[i-1][1]
            dice_per_organ.append(dice_score)
            hd95_per_organ.append(hd95_score)
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, dice_score, hd95_score))
        
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance:.4f} ± {std_dice:.4f}, mean_hd95: {mean_hd95:.4f} ± {std_hd95:.4f}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95, dice_per_organ, hd95_per_organ, std_dice, std_hd95


def val_one_epoch_simple(test_datasets,
                        test_loader,
                        model,
                        epoch, 
                        logger,
                        config):
    """简化的验证函数，只计算Dice、IoU、敏感性、特异性、ASD等指标，不保存图像"""
    
    def calculate_dice_coefficient(pred, gt):
        """计算Dice系数"""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt)
        if union == 0:
            return 1.0
        return 2.0 * intersection / union

    def calculate_iou(pred, gt):
        """计算IoU"""
        intersection = np.sum(pred * gt)
        union = np.sum(pred) + np.sum(gt) - intersection
        if union == 0:
            return 1.0
        return intersection / union

    def calculate_sensitivity(pred, gt):
        """计算敏感性(Sensitivity/Recall/TPR)"""
        tp = np.sum(pred * gt)
        fn = np.sum((1 - pred) * gt)
        if tp + fn == 0:
            return 1.0
        return tp / (tp + fn)

    def calculate_specificity(pred, gt):
        """计算特异性(Specificity/TNR)"""
        tn = np.sum((1 - pred) * (1 - gt))
        fp = np.sum(pred * (1 - gt))
        if tn + fp == 0:
            return 1.0
        return tn / (tn + fp)

    def calculate_asd(pred, gt, spacing=(1.0, 1.0, 1.0)):
        """计算平均表面距离(Average Surface Distance)"""
        from scipy import ndimage
        
        # 获取边界
        pred_boundary = pred - ndimage.binary_erosion(pred)
        gt_boundary = gt - ndimage.binary_erosion(gt)
        
        # 获取边界点坐标
        pred_coords = np.argwhere(pred_boundary)
        gt_coords = np.argwhere(gt_boundary)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return 0.0
        
        # 应用spacing
        pred_coords = pred_coords * np.array(spacing)
        gt_coords = gt_coords * np.array(spacing)
        
        # 计算从pred到gt的最短距离
        dist_pred_to_gt = []
        for point in pred_coords:
            distances = np.sqrt(np.sum((gt_coords - point) ** 2, axis=1))
            dist_pred_to_gt.append(np.min(distances))
        
        # 计算从gt到pred的最短距离
        dist_gt_to_pred = []
        for point in gt_coords:
            distances = np.sqrt(np.sum((pred_coords - point) ** 2, axis=1))
            dist_gt_to_pred.append(np.min(distances))
        
        # 计算平均表面距离
        all_distances = dist_pred_to_gt + dist_gt_to_pred
        return np.mean(all_distances)

    stime = time.time()
    model.eval()
    
    # 初始化指标累积器
    all_dice_scores = []
    all_iou_scores = []
    all_sensitivity_scores = []
    all_specificity_scores = []
    all_asd_scores = []
    
    organ_dice_scores = [[] for _ in range(config.num_classes - 1)]  # 排除背景
    organ_iou_scores = [[] for _ in range(config.num_classes - 1)]
    organ_sensitivity_scores = [[] for _ in range(config.num_classes - 1)]
    organ_specificity_scores = [[] for _ in range(config.num_classes - 1)]
    organ_asd_scores = [[] for _ in range(config.num_classes - 1)]
    
    with torch.no_grad():
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            # 获取numpy格式的数据
            img_np = img.squeeze(0).cpu().detach().numpy()
            msk_np = msk.squeeze(0).cpu().detach().numpy()
            
            # 生成预测结果
            if len(img_np.shape) == 3:  # 3D数据
                prediction_np = np.zeros_like(msk_np)
                
                # 逐slice预测
                for ind in range(img_np.shape[0]):
                    slice_img = img_np[ind, :, :]
                    x, y = slice_img.shape[0], slice_img.shape[1]
                    
                    # 调整尺寸
                    if x != config.input_size_h or y != config.input_size_w:
                        from scipy.ndimage import zoom
                        slice_img = zoom(slice_img, (config.input_size_h / x, config.input_size_w / y), order=3)
                    
                    # 根据网络类型处理输入通道数
                    if config.network.startswith('cumamba'):
                        # CUMamba系列模型需要3通道输入
                        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).float().cuda()  # (H, W) -> (1, H, W)
                        input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, H, W) -> (3, H, W) -> (1, 3, H, W)
                    else:
                        # VMUNet等其他模型使用1通道输入
                        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()  # (H, W) -> (1, 1, H, W)
                    
                    # 获取预测结果
                    outputs = model(input_tensor)
                    pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                    
                    # 调整回原尺寸
                    if x != config.input_size_h or y != config.input_size_w:
                        pred = zoom(pred, (x / config.input_size_h, y / config.input_size_w), order=0)
                    
                    prediction_np[ind] = pred
            else:
                # 2D数据处理
                x, y = img_np.shape[0], img_np.shape[1]
                if x != config.input_size_h or y != config.input_size_w:
                    from scipy.ndimage import zoom
                    img_np = zoom(img_np, (config.input_size_h / x, config.input_size_w / y), order=3)
                
                if config.network.startswith('cumamba'):
                    input_tensor = torch.from_numpy(img_np).unsqueeze(0).float().cuda()
                    input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)
                else:
                    input_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().cuda()
                
                outputs = model(input_tensor)
                prediction_np = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                
                if x != config.input_size_h or y != config.input_size_w:
                    prediction_np = zoom(prediction_np, (x / config.input_size_h, y / config.input_size_w), order=0)
            
            # 计算每个器官的指标
            case_dice_scores = []
            case_iou_scores = []
            case_sensitivity_scores = []
            case_specificity_scores = []
            case_asd_scores = []
            
            for class_idx in range(1, config.num_classes):  # 排除背景类别
                # 提取当前类别的二值mask
                pred_mask = (prediction_np == class_idx).astype(np.float32)
                gt_mask = (msk_np == class_idx).astype(np.float32)
                
                # 计算各项指标
                dice = calculate_dice_coefficient(pred_mask, gt_mask)
                iou = calculate_iou(pred_mask, gt_mask)
                sensitivity = calculate_sensitivity(pred_mask, gt_mask)
                specificity = calculate_specificity(pred_mask, gt_mask)
                
                try:
                    asd = calculate_asd(pred_mask, gt_mask)
                except:
                    asd = 0.0
                
                case_dice_scores.append(dice)
                case_iou_scores.append(iou)
                case_sensitivity_scores.append(sensitivity)
                case_specificity_scores.append(specificity)
                case_asd_scores.append(asd)
                
                # 添加到器官特定的累积器
                organ_dice_scores[class_idx - 1].append(dice)
                organ_iou_scores[class_idx - 1].append(iou)
                organ_sensitivity_scores[class_idx - 1].append(sensitivity)
                organ_specificity_scores[class_idx - 1].append(specificity)
                organ_asd_scores[class_idx - 1].append(asd)
            
            # 计算当前case的平均值
            case_avg_dice = np.mean(case_dice_scores)
            case_avg_iou = np.mean(case_iou_scores)
            case_avg_sensitivity = np.mean(case_sensitivity_scores)
            case_avg_specificity = np.mean(case_specificity_scores)
            case_avg_asd = np.mean(case_asd_scores)
            
            all_dice_scores.append(case_avg_dice)
            all_iou_scores.append(case_avg_iou)
            all_sensitivity_scores.append(case_avg_sensitivity)
            all_specificity_scores.append(case_avg_specificity)
            all_asd_scores.append(case_avg_asd)
            
            logger.info('idx %d case %s dice %f iou %f sen %f spe %f asd %f' % 
                       (i_batch, case_name, case_avg_dice, case_avg_iou, 
                        case_avg_sensitivity, case_avg_specificity, case_avg_asd))
            i_batch += 1
    
    # 计算总体平均值
    mean_dice = np.mean(all_dice_scores)
    mean_iou = np.mean(all_iou_scores)
    mean_sensitivity = np.mean(all_sensitivity_scores)
    mean_specificity = np.mean(all_specificity_scores)
    mean_asd = np.mean(all_asd_scores)
    
    # 计算每个器官的平均值
    dice_per_organ = [np.mean(scores) for scores in organ_dice_scores]
    iou_per_organ = [np.mean(scores) for scores in organ_iou_scores]
    sensitivity_per_organ = [np.mean(scores) for scores in organ_sensitivity_scores]
    specificity_per_organ = [np.mean(scores) for scores in organ_specificity_scores]
    asd_per_organ = [np.mean(scores) for scores in organ_asd_scores]
    
    etime = time.time()
    log_info = f'val epoch: {epoch}, mean_dice: {mean_dice:.4f}, mean_iou: {mean_iou:.4f}, mean_sen: {mean_sensitivity:.4f}, mean_spe: {mean_specificity:.4f}, mean_asd: {mean_asd:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    
    return mean_dice, mean_iou, mean_sensitivity, mean_specificity, mean_asd, dice_per_organ, iou_per_organ, sensitivity_per_organ, specificity_per_organ, asd_per_organ
