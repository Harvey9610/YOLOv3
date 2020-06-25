import torch
import torch.nn as nn
import numpy as np
import sys
import math

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def to_cpu(tensor):
    return tensor.detach().cpu()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.num_classes = 80
        self.input_size = 416
        self.anchors = [[[10,13], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]]]
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.num_anchors = 3
        self.ignore_thres = 0.5
        self.stride = [32, 16, 8]
        self.metrics = {}

        self.layer0 = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        ])

        self.nums = [1, 2, 8, 8, 4]
        self.in_features = [64, 128, 256, 512, 1024]

        self.layer1 = self.make_layers(self.nums[0], self.in_features[0])
        self.layer2 = self.make_layers(self.nums[1], self.in_features[1])
        self.layer3 = self.make_layers(self.nums[2], self.in_features[2])
        self.layer4 = self.make_layers(self.nums[3], self.in_features[3])
        self.layer5 = self.make_layers(self.nums[4], self.in_features[4])

        self.conv_group1 = self.conv_group(1024, 512)
        self.conv_group2 = self.conv_group(768, 256)
        self.conv_group3 = self.conv_group(384, 128)

        self.up_group1 = self.up_group(256)
        self.up_group2 = self.up_group(128)

        self.output_group1 = self.output_group(512)
        self.output_group2 = self.output_group(256)
        self.output_group3 = self.output_group(128)

    def make_layers(self, num, in_features):
        layers = nn.ModuleList([])

        conv3 = nn.Conv2d(in_features//2, in_features, kernel_size=3, stride=2, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(in_features)
        activation3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        layers += [conv3, bn3, activation3]

        for i in range(num):
            conv1 = nn.Conv2d(in_features, in_features//2, kernel_size=1, stride=1, bias=False)
            bn1 = nn.BatchNorm2d(in_features//2)
            activation1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            conv2 = nn.Conv2d(in_features//2, in_features, kernel_size=3, stride=1, padding=1, bias=False)
            bn2 = nn.BatchNorm2d(in_features)
            activation2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            layers += [conv1, bn1, activation1, conv2, bn2, activation2]

        return layers

    # [1024, 768, 384]
    # [512,  256, 128]
    def conv_group(self, add_features, in_features):
        layers = []

        conv1 = nn.Conv2d(add_features, in_features, kernel_size=1, stride=1, bias=False)
        bn1 = nn.BatchNorm2d(in_features)
        activation1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        conv2 = nn.Conv2d(in_features, 2*in_features, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(2*in_features)
        activation2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        conv3 = nn.Conv2d(2*in_features, in_features, kernel_size=1, stride=1, bias=False)
        bn3 = nn.BatchNorm2d(in_features)
        activation3 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        conv4 = nn.Conv2d(in_features, 2*in_features, kernel_size=3, stride=1, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(2*in_features)
        activation4 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        conv5 = nn.Conv2d(2*in_features, in_features, kernel_size=1, stride=1, bias=False)
        bn5 = nn.BatchNorm2d(in_features)
        activation5 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        layers += [conv1, bn1, activation1, conv2, bn2, activation2, conv3, bn3, activation3, conv4, bn4, activation4, conv5, bn5, activation5]
        return nn.ModuleList(layers)

    # [256, 128]
    def up_group(self, in_features):
        layers = []

        conv = nn.Conv2d(2*in_features, in_features, kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(in_features)
        activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        layers += [conv, bn, activation, upsample]
        return nn.ModuleList(layers)

    # [512, 256, 128]
    def output_group(self, in_features):
        layers = []

        conv1 = nn.Conv2d(in_features, 2*in_features, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d((2*in_features))
        activation1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        conv2 = nn.Conv2d(2*in_features, 3*(self.num_classes+5), kernel_size=1, stride=1)
        # activation='linear'

        layers += [conv1, bn1, activation1, conv2]
        return nn.ModuleList(layers)

    def predict_transform(self, prediction, anchors):
        batch_size = prediction.size(0)
        stride =  self.input_size // prediction.size(2)
        grid_size = self.input_size // stride
        bbox_attrs = 5 + self.num_classes

        prediction = prediction.view(batch_size, self.num_anchors, self.num_classes+5, grid_size, grid_size).contiguous()
        prediction = prediction.permute(0, 1, 3, 4, 2)
        # prediction.shape = [1, 3, 13, 13, 85]

        # sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[...,0] = torch.sigmoid(prediction[...,0])
        prediction[...,1] = torch.sigmoid(prediction[...,1])
        prediction[...,4] = torch.sigmoid(prediction[...,4])
        prediction[...,5:] = torch.sigmoid(prediction[...,5:])

        x = prediction[...,0].clone()
        y = prediction[...,1].clone()
        w = prediction[:,:,:,:,2].clone()
        h = prediction[:,:,:,:,3].clone()
        pred_conf = prediction[...,4].clone()
        pred_cls = prediction[...,5:].clone()

        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
        scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1)).contiguous()
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1)).contiguous()

        prediction[..., 0] = x.data + grid_x
        prediction[..., 1] = y.data + grid_y
        prediction[..., 2] = torch.exp(w.data) * anchor_w
        prediction[..., 3] = torch.exp(h.data) * anchor_h

        return prediction, x, y, w, h, pred_conf, pred_cls

    def build_targets(self, prediction, target, anchors, ignore_thres=0.5):
        # prediction = [1, 3*13*13, 85]
        batch_size = prediction.size(0)
        grid_size = int(math.sqrt(prediction.size(1) // self.num_anchors))
        stride = self.input_size // grid_size

        pred_boxes = prediction[:, :, :, :, :4]
        pred_conf = prediction[:, :, :, :, 4]
        pred_cls = prediction[:, :, :, :, 5:]

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask.float()
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        return tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask, class_mask, iou_scores


    def calculate_loss(self, x, y, w, h, pred_conf, pred_cls, tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask, class_mask, iou_scores):
        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        # print(f'x: {torch.max(x).item()}\t{torch.min(x).item()}')
        # print(f'y: {torch.max(x).item()}\t{torch.min(y).item()}')
        # print(f'w: {torch.max(x).item()}\t{torch.min(w).item()}')
        # print(f'h: {torch.max(x).item()}\t{torch.min(h).item()}')

        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

        # print(f'loss_x: {loss_x.item()}')
        # print(f'loss_y: {loss_y.item()}')
        # print(f'loss_w: {loss_w.item()}')
        # print(f'loss_h: {loss_h.item()}')

        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # print(f'loss_conf_obj: {loss_conf_obj}')
        # print(f'loss_conf_noobj: {loss_conf_noobj}')
        # print(f'loss_conf: {loss_conf}')
        # print(f'loss_cls: {loss_cls}')
        # print(f'total_loss: {total_loss}\n\n')

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        self.metrics = {
            "loss": to_cpu(total_loss).item(),
            "x": to_cpu(loss_x).item(),
            "y": to_cpu(loss_y).item(),
            "w": to_cpu(loss_w).item(),
            "h": to_cpu(loss_h).item(),
            "conf": to_cpu(loss_conf).item(),
            "cls": to_cpu(loss_cls).item(),
            "cls_acc": to_cpu(cls_acc).item(),
            "recall50": to_cpu(recall50).item(),
            "recall75": to_cpu(recall75).item(),
            "precision": to_cpu(precision).item(),
            "conf_obj": to_cpu(conf_obj).item(),
            "conf_noobj": to_cpu(conf_noobj).item(),
            # "grid_size": grid_size,
        }
        return total_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, weightfile='./weights/yolov3.weights'):
        # open the weights file
        fp = open(weightfile, 'rb')
        blocks = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.conv_group1, self.output_group1, self.up_group1, self.conv_group2, self.output_group2, self.up_group2, self.conv_group3, self.output_group3]
        n = [3, 9, 15, 51, 51, 27, 15, 4, 4, 15, 4, 4, 15, 4]

        # The first 5 values are header information
        # 1. Major version Number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        count = 0
        for i in range(14):
            j = 0
            if n[i] == 4 and isinstance(blocks[i][-1], nn.Conv2d):
                conv = blocks[i][0]
                bn = blocks[i][1]

                # get the number of weights of Batch Norm Layer
                num_bn_biases = bn.bias.numel()

                # load the bn_weights
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                # load the bn_bias
                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # load the bn_running_mean
                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # load the bn_running_var
                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                # copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)

                # get the number of bias of convolutional layer
                num_weights = conv.weight.numel()

                # load the conv_bias
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr  += num_weights

                # cast the loaded bias into dims of model bias.
                conv_weights = conv_weights.view_as(conv.weight.data)

                # copy the data to model
                conv.weight.data.copy_(conv_weights)

                # ################################################

                # get the number of bias of convolutional layer
                conv = blocks[i][3]
                num_biases = conv.bias.numel()

                # load the conv_bias
                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr += num_biases

                # cast the loaded bias into dims of model bias.
                conv_biases = conv_biases.view_as(conv.bias.data)

                # copy the data to model
                conv.bias.data.copy_(conv_biases)

                # get the number of bias of convolutional layer
                num_weights = conv.weight.numel()

                # load the conv_bias
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr  += num_weights

                # cast the loaded bias into dims of model bias.
                conv_weights = conv_weights.view_as(conv.weight.data)

                # copy the data to model
                conv.weight.data.copy_(conv_weights)
                count += 2

            elif n[i] == 4 and not isinstance(blocks[i][-1], nn.Conv2d):
                conv = blocks[i][0]
                bn = blocks[i][1]

                # get the number of weights of Batch Norm Layer
                num_bn_biases = bn.bias.numel()

                # load the bn_weights
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                # load the bn_bias
                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # load the bn_running_mean
                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # load the bn_running_var
                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr  += num_bn_biases

                # cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                # copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)

                # get the number of bias of convolutional layer
                num_weights = conv.weight.numel()

                # load the conv_bias
                conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                ptr  += num_weights

                # cast the loaded bias into dims of model bias.
                conv_weights = conv_weights.view_as(conv.weight.data)

                # copy the data to model
                conv.weight.data.copy_(conv_weights)

                count += 1

            else:
                while j < n[i]:
                    conv = blocks[i][j]
                    bn = blocks[i][j+1]

                    # get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # load the bn_weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # load the bn_bias
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # load the bn_running_mean
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # load the bn_running_var
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                    # get the number of bias of convolutional layer
                    num_weights = conv.weight.numel()

                    # load the conv_bias
                    conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
                    ptr  += num_weights

                    # cast the loaded bias into dims of model bias.
                    conv_weights = conv_weights.view_as(conv.weight.data)

                    # copy the data to model
                    conv.weight.data.copy_(conv_weights)

                    j += 3
                    count += 1

    def forward(self, x, targets = None):
        for i, k in enumerate(self.layer0):
            x = k(x)

        # layer1
        # for i, k in enumerate(self.layer1):
        #     x = k(x)

        l = [i for i in range(3, len(self.layer1)+1, 6)]

        for i in range(len(l)):
            if i:
                for j in range(l[i-1], l[i]):
                    x = self.layer1[j](x)

                x += identity
                identity = x
            else:
                for j in range(0, l[i]):
                    x = self.layer1[j](x)
                identity = x

        # layer2
        # for i, k in enumerate(self.layer2):
        #     x = k(x)

        l = [i for i in range(3, len(self.layer2)+1, 6)]

        for i in range(len(l)):
            if i:
                for j in range(l[i-1], l[i]):
                    x = self.layer2[j](x)

                x += identity
                identity = x
            else:
                for j in range(0, l[i]):
                    x = self.layer2[j](x)
                identity = x

        # layer3
        # for i, k in enumerate(self.layer3):
        #     x = k(x)

        l = [i for i in range(3, len(self.layer3)+1, 6)]

        for i in range(len(l)):
            if i:
                for j in range(l[i-1], l[i]):
                    x = self.layer3[j](x)

                x += identity
                identity = x
            else:
                for j in range(0, l[i]):
                    x = self.layer3[j](x)
                identity = x

        outputs_3 = x

        # layer4
        # for i, k in enumerate(self.layer4):
        #     x = k(x)

        l = [i for i in range(3, len(self.layer4)+1, 6)]

        for i in range(len(l)):
            if i:
                for j in range(l[i-1], l[i]):
                    x = self.layer4[j](x)

                x += identity
                identity = x
            else:
                for j in range(0, l[i]):
                    x = self.layer4[j](x)
                identity = x

        outputs_4 = x

        # layer5
        # for i, k in enumerate(self.layer5):
        #     x = k(x)

        l = [i for i in range(3, len(self.layer5)+1, 6)]

        for i in range(len(l)):
            if i:
                for j in range(l[i-1], l[i]):
                    x = self.layer5[j](x)

                x += identity
                identity = x
            else:
                for j in range(0, l[i]):
                    x = self.layer5[j](x)
                identity = x

        # return x

        # x = self.conv_group1(x)
        for i, k in enumerate(self.conv_group1):
            x = k(x)

        node1 = x

        # prediction_1 = self.output_group1(x)
        for i, k in enumerate(self.output_group1):
            x = k(x)

        prediction_1 = x
        x = node1

        # x = self.up_group1(x)
        for i, k in enumerate(self.up_group1):
            x = k(x)

        x = torch.cat((x, outputs_4), 1)

        # x = self.conv_group2(x)
        for i, k in enumerate(self.conv_group2):
            x = k(x)

        node2 = x

        # prediction_2 = self.output_group2(x)
        for i, k in enumerate(self.output_group2):
            x = k(x)

        prediction_2 = x
        x = node2

        # x = self.up_group2(x)
        for i, k in enumerate(self.up_group2):
            x = k(x)

        x = torch.cat((x, outputs_3), 1)

        # x = self.conv_group3(x)
        for i, k in enumerate(self.conv_group3):
            x = k(x)

        # prediction_3 = self.output_group3(x)
        for i, k in enumerate(self.output_group3):
            x = k(x)

        prediction_3 = x

        anchors_1 = [(116,90), (156,198), (373,326)]
        anchors_2 = [(30,61), (62,45), (59,119)]
        anchors_3 = [(10,13), (16,30), (33,23)]

        # prediction_1.shape = [1, 255, 13, 13]
        # result_1.shape = [1, 13*13*3, 85]

        result_1, x_1, y_1, w_1, h_1, pred_conf_1, pred_cls_1 = self.predict_transform(prediction_1, anchors_1)

        result_2, x_2, y_2, w_2, h_2, pred_conf_2, pred_cls_2 = self.predict_transform(prediction_2, anchors_2)

        result_3, x_3, y_3, w_3, h_3, pred_conf_3, pred_cls_3 = self.predict_transform(prediction_3, anchors_3)

        if targets == None:
            result_1[:, :, :, :, :4] = result_1[:, :, :, :, :4] * self.stride[0]
            result_2[:, :, :, :, :4] = result_2[:, :, :, :, :4] * self.stride[1]
            result_3[:, :, :, :, :4] = result_3[:, :, :, :, :4] * self.stride[2]
            result_1 = result_1.reshape(result_1.size(0), -1, self.num_classes+5)
            result_2 = result_2.reshape(result_2.size(0), -1, self.num_classes+5)
            result_3 = result_3.reshape(result_3.size(0), -1, self.num_classes+5)
            results = torch.cat([result_1, result_2], 1)
            results = torch.cat([results, result_3], 1)
            return results, 0

        else:
            tx_1, ty_1, tw_1, th_1, tconf_1, tcls_1, obj_mask_1, noobj_mask_1, class_mask_1, iou_scores_1 = self.build_targets(result_1, targets, anchors_1)
            tx_2, ty_2, tw_2, th_2, tconf_2, tcls_2, obj_mask_2, noobj_mask_2, class_mask_2, iou_scores_2 = self.build_targets(result_2, targets, anchors_2)
            tx_3, ty_3, tw_3, th_3, tconf_3, tcls_3, obj_mask_3, noobj_mask_3, class_mask_3, iou_scores_3 = self.build_targets(result_3, targets, anchors_3)

            loss_1 = self.calculate_loss(x_1, y_1, w_1, h_1, pred_conf_1, pred_cls_1, tx_1, ty_1, tw_1, th_1, tconf_1, tcls_1, obj_mask_1, noobj_mask_1, class_mask_1, iou_scores_1)
            loss_2 = self.calculate_loss(x_2, y_2, w_2, h_2, pred_conf_2, pred_cls_2, tx_2, ty_2, tw_2, th_2, tconf_2, tcls_2, obj_mask_2, noobj_mask_2, class_mask_2, iou_scores_2)
            loss_3 = self.calculate_loss(x_3, y_3, w_3, h_3, pred_conf_3, pred_cls_3, tx_3, ty_3, tw_3, th_3, tconf_3, tcls_3, obj_mask_3, noobj_mask_3, class_mask_3, iou_scores_3)

            loss = loss_1 + loss_2 + loss_3

            result_1 = result_1.reshape(result_1.size(0), -1, self.num_classes+5)
            result_2 = result_2.reshape(result_2.size(0), -1, self.num_classes+5)
            result_3 = result_3.reshape(result_3.size(0), -1, self.num_classes+5)
            results = torch.cat([result_1, result_2], 1)
            results = torch.cat([results, result_3], 1)
            return results, loss
