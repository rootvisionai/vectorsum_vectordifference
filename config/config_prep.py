from yacs.config import CfgNode as cnNode
import json

# -----------------------------------------------------------------------------
# Verimedi Default Config definition
# -----------------------------------------------------------------------------

with open('./config/config.json','r') as f:
    cj = json.loads(f.read())
    
cn        = cnNode(cj)
cn.resume = '/model_{}/dataset@{}_arch@{}_loss@{}_embedsize@{}_alpha@{}_margin@{}_optimizer@{}_batch@{}_decomposition@{}'.format(
            cn.dataset,cn.dataset,
            cn.model,cn.loss,
            int(cn.embedding_size),
            cn.alpha, cn.mrg, 
            cn.optimizer, 
            int(cn.batch_size),
            int(cn.decomposition)).replace('.','')
'''
cn                = cnNode()
cn.name           = "default"
cn.dataset        = "v4"           #v1, v2, jopnet_fastrcnn, jopnet_maskrcnn, nlm_stylized'
cn.embedding_size = 512            #64,128,256,512
cn.pc_size        = 0
cn.batch_size     = 64             #16,32,64,128
cn.input_size     = 448
cn.debug_images   = '../debug_images'
cn.epochs         = 30             #20,40,60
cn.device         = "cpu"          #cpu,cuda
cn.num_workers    = 0              #0,6,8
cn.model          = "resnet50"     #resnet18, resnet50, resnet101, googlenet, bn_inception
cn.loss           = "ProxyAnchor"  #ProxyNCA,ProxyAnchor,MS,Contrastive,Triplet,NPair
cn.optimizer      = "adamw"        #sgd,adam,rmsprop,adamw
cn.lr             = 0.0001 
cn.weight_decay   = 0.0001
cn.lr_decay_step  = 10
cn.lr_decay_gamma = 0.5
cn.alpha          = 32
cn.mrg            = 0.5            #0.1<=
cn.img_per_class  = 0
cn.bn_freeze      = 0
cn.l2_norm        = 1
cn.epoch_interval = 1
cn.remark         = 0
cn.models_dir     = "../recognition_models"
cn.resume         = '/model_{}/dataset@{}_arch@{}_loss@{}_embedsize@{}_alpha@{}_margin@{}_optimizer@{}_batch@{}_{}'.format(
                    cn.dataset,cn.dataset,
                    cn.model,cn.loss,
                    cn.embedding_size,
                    cn.alpha, cn.mrg, 
                    cn.optimizer, 
                    cn.batch_size, 
                    cn.remark).replace('.','')
'''