import data,util,torch
from yolo.loss import Loss_yolov1
from torchvision import models
import torchvision.transforms as transforms
from yolo.network import YOLOv1
from yolo.yolo_resnet import YOLOv1_Resnet
from yolo.yolo_vgg import vgg19,vgg19_bn
from torch.autograd import Variable
import warnings,os
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()

train_transformer = [
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(448,scale=(0.7,1.0)),
    transforms.ColorJitter(brightness=1, contrast=1.5, saturation=1.5, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
test_transformer = [
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


def t(load_path=None,fronzen=True,offset=0):
    epoch = 1000
    lr = 0.01


    train_dataset = data.voc_dataset('train',transform=train_transformer)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=False,num_workers=8)
    test_dataset = data.voc_dataset('test', transform=test_transformer)
    test_loader = data.DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=8)


    # model = YOLOv1_Resnet()
    # model=YOLOv1()
    model=vgg19_bn()
    if use_gpu:
        model.cuda()
    if load_path!=None:
        try:
            model.load_state_dict(torch.load(load_path))
        except Exception:
            print('model not Found')
    if not fronzen:
        for para in model.features.parameters():
            para.requires_grad = True
    loss_func = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    torch.autograd.set_detect_anomaly(True)

    best_eval_loss=torch.Tensor([8]).cuda() if use_gpu else torch.Tensor([8])
    for e in range(epoch):
        epoch_loss =torch.Tensor([0]).cuda() if use_gpu else torch.Tensor([0])
        epoch_eval_loss = torch.Tensor([0]).cuda() if use_gpu else torch.Tensor([0])
        epoch_part_loss=torch.zeros(4).cuda() if use_gpu else torch.zeros(4)
        if e == 0:
            lr = 0.001
        if e == 5:
            lr = 0.01
        if e == 80:
            lr = 0.001
        if e == 110:
            lr = 0.0001
        if e in [0,5,80,110]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if e < offset: continue
        print('\n\nStarting epoch %d / %d' % (e + 1, epoch))
        print('Learning Rate for this epoch: {}'.format(lr))

        model.train()
        for i,(inputs,target,_) in enumerate(train_loader):
            inputs = Variable(inputs)
            target = Variable(target)
            if use_gpu:
                inputs, target = inputs.cuda(), target.cuda()
            pred = model(inputs)
            loss,loc_loss,conf_loss_obj,conf_loss_no_obj,cls_loss = loss_func(pred, target)

            epoch_loss = epoch_loss + loss
            part_loss = torch.Tensor([loc_loss, conf_loss_obj, conf_loss_no_obj, cls_loss])/test_loader.batch_size
            if use_gpu:
                part_loss = part_loss.cuda()
            epoch_part_loss = epoch_part_loss + part_loss

            if i%20==0:
                print("Epoch %d/%d| Step %d/%d Loss: %.2f ,loc_loss : %.2f,conf_loss_obj: %.2f , conf_loss_no_obj: %.2f, cls_loss: %.2f" % (e + 1, epoch, i + 1, len(train_loader), loss,part_loss[0],part_loss[1],part_loss[2],part_loss[3]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_part_loss =epoch_part_loss / len(test_loader)
        print("Train Epoch %d/%d| TrainMeanLoss: %.2f ,loc_loss : %.2f, conf_loss_obj: %.2f , conf_loss_no_obj: %.2f, cls_loss: %.2f " % (e + 1, epoch, epoch_loss/len(train_loader),epoch_part_loss[0],epoch_part_loss[1],epoch_part_loss[2],epoch_part_loss[3]))

        model.eval()
        with torch.no_grad():
            for i, (inputs, target,_) in enumerate(test_loader):
                inputs = Variable(inputs)
                target = Variable(target)
                if use_gpu:
                    inputs, target = inputs.cuda(), target.cuda()
                    pred = model(inputs)
                    loss,loc_loss,conf_loss_obj,conf_loss_no_obj,cls_loss = loss_func(pred, target)
                    epoch_eval_loss = epoch_eval_loss + loss
            eval_mean_Loss=epoch_eval_loss/len(test_loader)
            print('Eval Epoch %d/%d| EvalMeanLoss : %.2f' % (e + 1, epoch, eval_mean_Loss))
            if eval_mean_Loss<best_eval_loss and e>10:
                best_eval_loss = eval_mean_Loss
                torch.save(model.state_dict(), './model/YOLOv1_ce_sigmoid_Fronzen_best.pth')
                print('best model Saved')





if __name__ == '__main__':
    print('ceSigmoid fronzen=True,offset=0')
    t(load_path='./model/YOLOv1_ce_sigmoid_Fronzen_best.pth',fronzen=True,offset=0)