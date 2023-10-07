python3 tools/test.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.IF_WITH_CENTER "('no')" MODEL.NAME "('HRNet32')" MODEL.PRETRAIN_PATH "('/home/fzl/Projects/ISP-reID/pretained/hrnetv2_w32_imagenet_pretrained.pth')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/fzl/Projects/HJL-re-id/MDRSREID/Dataset/market1501')" CLUSTERING.PART_NUM "(5)" CLUSTERING.ENHANCED "(False)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-ISP-5')"  OUTPUT_DIR "('/home/fzl/Projects/ISP-reID-master-2/log_final/ISP_market/ISP-market-latent-parsingloss*0.3-5-arm')" TEST.WITH_ARM "(True)" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('/home/fzl/Projects/ISP-reID-master-2/log_final/ISP_market/ISP-market-latent-parsingloss*0.3-5/HRNet32_model_120.pth')"
