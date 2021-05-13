import os
import datetime
import argparse
import numpy

parser = argparse.ArgumentParser(description='Flow-Directed Interpolation Kernel')

parser.add_argument('--matrix_row', type=int, default=140, help='matrix row size')
parser.add_argument('--matrix_col', type=int, default=140, help='matrix col size')
#parser.add_argument('--dataset_dir',nargs='+', default="/raid/data/ml-powergrid/Auto-keras/org_dataset_100k",help = 'the path of selected datasets')
#the dataset are generated in PASA ALPHA: ~/LLVM-Tracer/DDDG_gen/NPB-BENCHMARRK
#parser.add_argument('--dataset_dir',nargs='+', default="/raid/data/ml-powergrid/Auto-keras/matrix_100_dataset_100k",help = 'the path of selected datasets')
#parser.add_argument('--dataset_dir',nargs='+', default="/raid/data/ml-powergrid/Auto-keras/matrix_cg_size_s_dataset_1k",help = 'the path of selected datasets')

parser.add_argument('--data_dir',type = str, default='/cc/home/AutoHPCnet-benchmark/', help='dataset source directory')

parser.add_argument('--benchmark',type = str, default='AMG', help ='bencmark type')
parser.add_argument('--searchType',type = str, default='fullInput', help ='model search type')

#bayesian parameters
parser.add_argument('--bayesian_initial_samples', type=int, default=50, help='samples for bayesian algorithm')
parser.add_argument('--bf_samples', type=int, default=50, help='samples for visialize the black box function')
parser.add_argument('--sample_size', type=int, default=320, help='samples for NN test')
parser.add_argument('--FR_ratio', type=float, default=1.0, help='the best feature reduction ratio after searching')

#autokeras parameters
parser.add_argument('--train_samples', type=int, default=80000, help='sample size for training data')
parser.add_argument('--test_samples', type=int, default=20000, help='sample size for test data')
parser.add_argument('--numEpoch', '-e', type = int, default=500, help= 'Number of epochs to train(default:1000)')

parser.add_argument('--debug',action = 'store_true', help='Enable debug mode')
parser.add_argument('--preprocessing',type = str, default=None, help ='processing method')

parser.add_argument('--TRAIN_SET_RATIO', type = int, default=80, help = 'Split a dataset into trainining and validation by percentage (default: 97)')
parser.add_argument('--single_output', type = int, default=97,
                    help='output single frame or not (multiple frame)')
parser.add_argument('--task', default='interp_blur',
                    choices= ['interp', 'interp_blur'],
                    help= 'specify tasks: interp or interp_blur')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--model_index', type=int, default=0, help='index for different models ')
parser.add_argument('--save_every_n_epoch', type = int, default=50, help= 'size for training steps')

parser.add_argument('--BATCH_SIZE', '-b', type = int ,default=10, help = 'batch size (default:10)' )
# parser.add_argument('--workers', '-w', type =int,default=1, help = 'parallel workers for loading training samples (default : 1.6*10 = 16)')
# parser.add_argument('--channels', '-c', type=int,default=3, choices = [1,3], help ='channels of images (default:3)')
parser.add_argument('--filter_size', '-f', type=int, default=4, help = 'the size of filters used (default: 4)',
                    choices=[2,4,6,5,51]
                    )

parser.add_argument('--gradclip', type =float, default= 5.0, help= 'gradient clipping (default: 5.0)')
parser.add_argument('--lr', type =float, default=0.001, help= 'the basic learning rate for three subnetworks (default: 0.002)')
parser.add_argument('--rectify_lr', type=float, default=0.001, help  = 'the learning rate for rectify/refine subnetworks (default: 0.001)')

parser.add_argument('--save_which', '-s', type=int, default=0, choices=[0,1], help='choose which result to save: 0 ==> interpolated, 1==> rectified')
parser.add_argument('--epsilon', type = float, default=1e-9, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--weight_decay', type = float, default=0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type=int, default=20, help = 'the patience of reduce on plateou')
parser.add_argument('--factor', type = float, default=0.8, help = 'the factor of reduce on plateou')


#default=None, help ='path to the pretrained model weights')
parser.add_argument('--pretrained', dest='SAVED_Bayesian_MODEL', default=None, help ='path to the pretrained model weights')
#parser.add_argument('--pretrained2', dest='SAVED_MODEL2', default=None, help ='path to the pretrained model weights')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder' )
parser.add_argument('--use_cuda', default=True, type = bool, help='use cuda or not')
parser.add_argument('--use_cudnn',default=1,type=int, help = 'use cudnn or not')
# parser.add_argument('--nocudnn', dest='use_cudnn', default=)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

parser.add_argument('--image_pair_dir', dest='image_pair_dir', default=None, help ='dir to the test sample')
parser.add_argument('--uid', type=str, default= None, help='unique id for the training')
parser.add_argument('--force', action='store_true', help='force to override the given uid')

parser.add_argument('--modeltype',type = str, default= None, help = 'the model type and inputs chosen')
parser.add_argument('--withsimilarcase',type = int, default= 0, help = 'input data with similar case solution')

args = parser.parse_args()

if args.uid == None:
    unique_id = str(numpy.random.randint(0, 100000))
#    unique_id = args.modeltype +'-'+ args.preprocessing + '-sim'+str(args.withsimilarcase)
    print("revise the unique id to a random numer " + str(unique_id))
    args.uid = unique_id
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
    save_path = './model_weights/'+ args.uid  +'-' + timestamp
    save_bayesian_path = './Bayesian_model_weights/' + args.uid + '-' + timestamp
else:
    if args.uid is not list:
        save_path = './model_weights/'+ str(args.uid) + '/'
        save_bayesian_path = './Bayesian_model_weights/'+ str(args.uid) + '/'
    else:
        save_paths =[]
# save_path =os.path.join( os.getcwd(), save_path)
# print("no pth here : " + save_path + "/best"+".pth")
if not os.path.exists(save_path + "/best"+".pth"):
    # print("no pth here : " + save_path + "/best" + ".pth")
    os.makedirs(save_path,exist_ok=True)
else:
    if not args.force:
        raise("please use another uid ")
    else:
        print("override this uid" + args.uid)

if not os.path.exists(save_bayesian_path + "/logs"+".json"):
    # print("no pth here : " + save_path + "/best" + ".pth")
    os.makedirs(save_bayesian_path,exist_ok=True)
else:
    if not args.force:
        raise("please use another uid ")
    else:
        print("override this uid" + args.uid)

parser.add_argument('--save_path',default=save_path,help = 'the output dir of weights')
parser.add_argument('--save_bayesian_path',default=save_bayesian_path,help = 'the output dir of bayesian model')
parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()

with open(args.log, 'w') as f:
    f.close()
with open(args.arg, 'w') as f:
    print(args)
    print(args,file=f)
    f.close()

model_index = 0
