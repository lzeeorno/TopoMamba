from datasets.dataset import Synapse_dataset
from losses import CombinedSegTopologyLoss
from utils import CeDiceLoss


class setting_config:
    """Synapse training config aligned to the TopoMamba paper."""

    datasets_name = "synapse"
    network = "TopoMamba_3D_t"
    num_classes = 9
    input_channels = 3
    input_size_h = 512
    input_size_w = 512

    vmunet_config = {
        "num_classes": num_classes,
        "input_channels": input_channels,
        "model_name": "vmunet_s",
        "depths": [2, 2, 2, 2],
        "depths_decoder": [2, 2, 2, 1],
        "drop_path_rate": 0.2,
        "load_ckpt_path": None,
        "pretrained_path": "pre_trained_weights/vmamba_small_e238_ema.pth",
        "load_pretrained": True,
    }

    topomamba_2d_t_config = {
        "num_classes": num_classes,
        "input_channels": input_channels,
        "model_name": "TopoMamba_2D_t",
        "load_pretrained": True,
        "pretrained_path": "pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth",
        "depths": [2, 2, 5, 2],
        "depths_decoder": [2, 5, 2, 2],
        "dims": [96, 192, 384, 768],
        "dims_decoder": [768, 384, 192, 96],
        "drop_path_rate": 0.1,
        "d_state": 1,
        "expand": 1.0,
        "branch_mode": "both",
        "enable_cache": True,
        "enable_gating": True,
        "hsic_proj_dim": 32,
        "hsic_alpha": 0.5,
        "hsic_temperature": 1.5,
        "hsic_residual": 0.2,
        "scan_cache_mode": "buffer",
        "fusion_method": "hsic",
    }
    topomamba_2d_s_config = {
        **topomamba_2d_t_config,
        "model_name": "TopoMamba_2D_s",
        "pretrained_path": "pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth",
        "depths": [2, 2, 8, 2],
        "depths_decoder": [2, 8, 2, 2],
        "drop_path_rate": 0.1,
        "expand": 2.0,
        "hsic_proj_dim": 64,
    }
    topomamba_2d_b_config = {
        **topomamba_2d_s_config,
        "model_name": "TopoMamba_2D_b",
        "pretrained_path": "pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth",
        "depths": [2, 2, 15, 2],
        "depths_decoder": [2, 15, 2, 2],
        "dims": [128, 256, 512, 1024],
        "dims_decoder": [1024, 512, 256, 128],
        "drop_path_rate": 0.2,
        "hsic_proj_dim": 96,
    }

    # Compatibility name for older local scripts; not used as the default.
    cumamba_t_config = topomamba_2d_t_config
    cumamba_s_config = topomamba_2d_s_config
    cumamba_b_config = topomamba_2d_b_config

    topomamba_3d_t_config = {
        "num_classes": num_classes,
        "input_channels": 1,
        "model_name": "TopoMamba_3D_t",
        "backbone_in_chans": 4,
        "backbone_out_chans": 4,
        "dims": [48, 96, 192, 384],
        "depths": [2, 2, 2, 2],
        "hidden_size": 768,
        "hsic_proj_dim": 32,
        "hsic_alpha": 0.8,
        "hsic_temperature": 1.5,
        "hsic_residual": 0.2,
        "enable_cache": True,
        "pretrained_path": "pre_trained_weights/tmp_model_ep799_0.8498.pt",
        "load_pretrained": True,
    }
    synapse3d_training_mode = "3d_fullres"
    synapse3d_planner = "auto"
    synapse3d_target_patch = (128, 128, 128)
    synapse3d_batch_size = 2
    synapse3d_max_batch_size = 4
    synapse3d_epochs = 1000
    synapse3d_steps_per_epoch = 250
    synapse3d_patch_candidates = (
        (128, 128, 128),
        (96, 128, 128),
        (64, 128, 128),
        (48, 128, 128),
        (32, 128, 128),
        (32, 96, 96),
    )

    _config_map = {
        "vmunet": vmunet_config,
        "TopoMamba_2D_t": topomamba_2d_t_config,
        "TopoMamba_2D_s": topomamba_2d_s_config,
        "TopoMamba_2D_b": topomamba_2d_b_config,
        "topomamba_2d_t": topomamba_2d_t_config,
        "topomamba_2d_s": topomamba_2d_s_config,
        "topomamba_2d_b": topomamba_2d_b_config,
        "CUMamba_t": topomamba_2d_t_config,
        "CUMamba_s": topomamba_2d_s_config,
        "CUMamba_b": topomamba_2d_b_config,
        "cumamba_t": topomamba_2d_t_config,
        "cumamba_s": topomamba_2d_s_config,
        "cumamba_b": topomamba_2d_b_config,
        "cumamba_v3_t": topomamba_2d_t_config,
        "cumamba_v3_s": topomamba_2d_s_config,
        "cumamba_v3_b": topomamba_2d_b_config,
        "dzzmamba_t": topomamba_2d_t_config,
        "dzzmamba_s": topomamba_2d_s_config,
        "dzzmamba_b": topomamba_2d_b_config,
        "TopoMamba_3D_t": topomamba_3d_t_config,
        "topomamba_3d_t": topomamba_3d_t_config,
    }
    model_config = _config_map[network]

    data_path = "./data/Synapse/train_npz/"
    list_dir = "./data/Synapse/lists/lists_Synapse/"
    volume_path = "./data/Synapse/test_vol_h5/"
    datasets = Synapse_dataset
    z_spacing = 1

    topology_loss_enabled = True
    topology_loss_weight = 0.05
    topology_loss_max_elements = 65536
    topology_critical_weight = 4.0
    topology_focal_gamma = 2.0
    topology_foreground_classes = tuple(range(1, num_classes))
    loss_weight = [1, 1]
    criterion = CombinedSegTopologyLoss(
        CeDiceLoss(num_classes, loss_weight),
        num_classes=num_classes,
        enabled=topology_loss_enabled,
        topology_weight=topology_loss_weight,
        focal_gamma=topology_focal_gamma,
        critical_weight=topology_critical_weight,
        foreground_classes=topology_foreground_classes,
        max_elements=topology_loss_max_elements,
    )

    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 2050
    world_size = None
    rank = None
    amp = False
    use_compile = False
    pin_memory = True
    resume_training = True
    gradient_clip_norm = 12.0
    accum_steps = 1

    batch_size = 5
    epochs = 300
    work_dir = f"results/{network}_{datasets_name}/"
    print_interval = 20
    val_interval = 20
    save_interval = 100
    test_weights_path = f"results/{network}_{datasets_name}/checkpoints/best.pth"
    threshold = 0.5

    nnunet_light_preprocess = True
    ct_hu_clip = (-125.0, 275.0)
    ct_roi_crop = False
    postprocess_enabled = False
    postprocess_keep_largest_component = False
    postprocess_fill_holes = False

    opt = "AdamW"
    lr = 3e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0.01
    amsgrad = False
    sch = "CosineAnnealingLR"
    T_max = epochs
    eta_min = 6e-7
    last_epoch = -1
    warm_up_epochs = 10
    warm_lr_start = 1e-6
    warm_lr_end = lr
    warm_lr_step = (warm_lr_end - warm_lr_start) / warm_up_epochs
