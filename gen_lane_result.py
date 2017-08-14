import os
import cv2
import logging

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn.modules import Softmax2d

import getmodel
import datasets
from datasets import img_transforms
from cmd_args import parser

# logger
logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)


def main():
    global args
    args = parser.parse_args()

    # create model
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("   ARCH: {}".format(args.arch))
    logger.info("    NET: {}".format(args.net))
    logger.info("Classes: {}".format(args.classes))
    model = getmodel.GetModel(args)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True
    softmax_2d = Softmax2d()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            logger.fatal("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger.fatal("args.resume needed.")

    # Data loading code
    mean = [103.939, 116.779, 123.68]
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageName(args.dataroot, args.testlist, img_transforms.Compose([
            img_transforms.Resize([737, 289]),
            img_transforms.ToTensor(),
            img_transforms.Normalize(mean=mean)
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not (args.savedir or args.savedir_prob):
        logger.fatal("args.savedir or args.savedir_prob needed.")

    logger.info('>>>>>>>>>>>>>>>> Start >>>>>>>>>>>>>>>>')
    model.eval()
    for i, (inputs, names) in enumerate(test_loader):
        inputs = inputs.cuda(async=True)
        input_var = torch.autograd.Variable(inputs, volatile=True)
        output_var = model(input_var)
        output_var = softmax_2d(output_var)
        output = output_var.data
        bs = output.size(0)
        _, pred = output.max(1)
        pred = pred.cpu().numpy().astype('uint8')
        output = output * 255.0
        output = output.cpu().numpy().astype('uint8')
        for j in range(bs):
            name = names[j]
            name = name[:name.rfind('.')]
            assert isinstance(name, str)
            if args.savedir:
                pred_label = pred[j, :, :, :]
                assert pred_label.ndim == 3
                pred_label = pred_label.transpose((1, 2, 0))
                pred_label_name = os.path.join(args.savedir, name)
                pred_label_dir = os.path.dirname(pred_label_name)
                if not os.path.isdir(pred_label_dir):
                    os.makedirs(pred_label_dir)
                cv2.imwrite(pred_label_name + '.png', pred_label)

            if args.savedir_prob:
                pred_prob = output[j, :, :, :]
                assert pred_prob.ndim == 3
                assert pred_prob.shape[0] == 5
                pred_prob_name = os.path.join(args.savedir_prob, name)
                pred_prob_dir = os.path.dirname(pred_prob_name)
                if not os.path.isdir(pred_prob_dir):
                    os.makedirs(pred_prob_dir)
                for c in range(1, 5):
                    cv2.imwrite(pred_prob_name + '_' + str(c) + '.png', pred_prob[c, :, :])
        if (i + 1) % args.print_freq == 0:
            logger.info('Iteration {} done.'.format(i + 1))


if __name__ == '__main__':
    main()
