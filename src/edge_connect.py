import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from scipy.misc import imread,imresize

import torch.nn as nn
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as F
import random
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, config.TEST_SEMANTIC_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, config.TRAIN_SEMANTIC_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, config.VAL_SEMANTIC_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self, iter_num):
        if self.config.MODEL == 1:
            self.edge_model.save(iter_num)

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save(iter_num)

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            sample_time = 20
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_gray, edges, masks, semantics = self.cuda(*items)

                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration

                    # visualize the output of the network
                    if iteration % self.config.SAMPLE_INTERVAL == 0:
                        out_file = []
                        for i in range(4):
                            image_out = outputs[i].detach().cpu().numpy()
                            image_gray_input = images_gray[i].detach().cpu().numpy()
                            canny_edge = edges[i].detach().cpu().numpy()
                            mask_output = masks[i].detach().cpu().numpy()

                            image_out = np.floor(image_out.transpose(1,2,0)*255)
                            image_gray_input = np.floor(image_gray_input.transpose(1,2,0)*255)
                            canny_edge = np.floor(canny_edge.transpose(1,2,0)*255)
                            mask_output = np.floor(mask_output.transpose(1,2,0)*255)

                            #print(image.shape)
                            out = np.concatenate((image_out, image_gray_input, canny_edge, mask_output), axis = 1)
                            out_file.append(out)
                        final = np.concatenate((out_file[0], out_file[1], out_file[2], out_file[3]), axis = 0)
                        cv.imwrite(os.path.join(self.config.PATH,"edge_samples", str(iteration)+"_edge_model_output.png"), final)

                    #print("output_shape:{}".format(outputs.shape))


                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks, semantics)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    if iteration % self.config.SAMPLE_INTERVAL == 0:
                        out_file = []
                        for i in range(4):
                            image_out = outputs[i].detach().cpu().numpy()
                            img_input = images[i].detach().cpu().numpy()
                            #image_gray_input = images_gray[i].detach().cpu().numpy()
                            canny_edge = edges[i].detach().cpu().numpy()
                            mask_output = masks[i].detach().cpu().numpy()
                            semantic_output = semantics[i].detach().cpu().numpy()

                            image_out = np.floor(image_out.transpose(1,2,0)*255)
                            img_input = np.floor(img_input.transpose(1,2,0)*255)
                            #image_gray_input = np.floor(image_gray_input.transpose(1,2,0)*255)
                            canny_edge = np.floor(canny_edge.transpose(1,2,0)*255)
                            mask_output = np.floor(mask_output.transpose(1,2,0)*255)
                            semantic_output = np.floor(semantic_output.transpose(1,2,0)*255)

                            # to make sure the channels should be the compatible
                            canny_edge = np.concatenate((canny_edge, canny_edge, canny_edge), axis = 2)
                            mask_output = np.concatenate((mask_output, mask_output, mask_output), axis = 2)
                            semantic_output = np.concatenate((semantic_output, semantic_output, semantic_output), axis = 2)

                            #print(image.shape)
                            out = np.concatenate((image_out, img_input, canny_edge, semantic_output, mask_output), axis = 1)
                            out_file.append(out)
                        final = np.concatenate((out_file[0], out_file[1], out_file[2], out_file[3]), axis = 0)
                        cv.imwrite(os.path.join(self.config.PATH,"inpainting_samples", str(iteration)+"_inpainting_model_output.png"), final)

                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks, semantics)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                # joint model
                else:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration

                if iteration % sample_time == 0:
                    print("Time to sampple the result...")

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    #print("Begin to sample the checkpoints")
                    #image = outputs[0].cpu()
                    pass
                    #print("Output's shape:{}".format(outputs.shape))

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save(iteration)

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def paddingtoFour(self, data):
        h = data.shape[2]
        w = data.shape[3]
        padding_h = 0 if h%4==0 else 4-h%4
        padding_w = 0 if w%4==0 else 4-w%4
        padding_cal = nn.ReflectionPad2d((padding_w,0,padding_h,0))
        img_t = padding_cal(data)
        return img_t

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        print(len(test_loader))

        index = 0
        with torch.no_grad():
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                images, images_gray, edges, masks, semantics = self.cuda(*items)

                images = self.paddingtoFour(images)
                images_gray = self.paddingtoFour(images_gray)
                edges = self.paddingtoFour(edges)
                masks = self.paddingtoFour(masks)
                semantics = self.paddingtoFour(semantics)
                #print(images.shape)
                #print(images_gray.shape)
                #print(edges.shape)
                #print(masks.shape)
                #print(semantics.shape)

                index += 1

                # edge model
                if model == 1:
                    outputs = self.edge_model(images_gray, edges, masks)
                    outputs_merged = (outputs * masks) + (edges * (1 - masks))

                # inpaint model
                elif model == 2:
                    outputs = self.inpaint_model(images, edges, masks, semantics)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                # inpaint with edge model / joint model
                else:
                    if self.detectBox(masks):
                        print("it's a box")
                        outputs=self.resize_to_merge(images,edges,masks,images_gray, semantics)
                    elif images.shape[2]<1024 or images.shape[3]<1024:
                        outputs=self.ensemble(images,masks,edges,images_gray, semantics)#picture,picture_mask,picture_edges,picture_gray
                    else:
                        #complete your work for other mask 
                        outputs=self.crop_to_merge(images,edges,masks,images_gray, semantics)
                        print("other mask")
                    # outputs,outputs_mask=images,masks
                    # for i in range(1):
                    #     outputs,outputs_mask=self.splitimages(outputs,outputs_mask,edges,images_gray)
                    #edges= self.edge_model(images_gray, edges, masks).detach()
                    # outputs = self.inpaint_model(images, edges, masks)
                    # outputs=self.ensemble(images,masks,edges,images_gray)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                output = self.postprocess(outputs_merged)[0]
                #output = self.postprocess(outputs)[0]
                path = os.path.join(self.results_path, name)
                print(index, name)

                imsave(output, path)

                torch.cuda.empty_cache()

                if self.debug:
                    edges = self.postprocess(1 - edges)[0]
                    masked = self.postprocess(images * (1 - masks) + masks)[0]
                    fname, fext = name.split('.')

                    imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                    imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        sample_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        #items = next(self.sample_iterator)
        print(len(sample_loader))
        for items in sample_loader:
            images, images_gray, edges, masks, semantics = self.cuda(*items)

            # edge model
            if model == 1:
                iteration = self.edge_model.iteration
                inputs = (images_gray * (1 - masks)) + masks
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                iteration = self.inpaint_model.iteration
                inputs = (images * (1 - masks)) + masks
                outputs = self.inpaint_model(images, edges, masks, semantics)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                iteration = self.inpaint_model.iteration
                inputs = (images * (1 - masks)) + masks
                outputs = self.edge_model(images_gray, edges, masks).detach()
                edges = (outputs * masks + edges * (1 - masks)).detach()
                outputs = self.inpaint_model(images, edges, masks, semantics)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            if it is not None:
                iteration = it

            image_per_row = 2
            if self.config.SAMPLE_SIZE <= 6:
                image_per_row = 1

            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs),
                self.postprocess(outputs_merged),
                img_per_row = image_per_row
            )


            path = os.path.join(self.samples_path, self.model_name)
            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def resize_to_merge(self, img, edges, masks,gray, semantics):
        img=self.postprocess(img)[0].cpu().numpy()
        masks=masks[0,0,:,:].cpu().numpy()
        edges=edges[0,0,:,:].cpu().numpy()
        gray=gray[0,0,:,:].cpu().numpy()
        semantics=semantics[0,0,:,:].cpu().numpy()

        imgh, imgw = img.shape[0:2]
        #Processing 128 size
        img128 = imresize(img, [128, 128])
        edges128 = imresize(edges, [128, 128])
        masks128 = imresize(masks, [128, 128])
        grays128 = imresize(gray, [128, 128])
        masks128 = (masks128 > 0).astype(np.uint8) * 255
        semantics128 = imresize(semantics, [128, 128])

        #Convert 128 size numpy array back to tensor
        grays128=self.to_tensor(grays128)[np.newaxis,:].cuda()
        edges128=self.to_tensor(edges128)[np.newaxis,:].cuda()
        img128=self.to_tensor(img128.astype(np.uint8))[np.newaxis,:].cuda()
        masks128=self.to_tensor(masks128)[np.newaxis,:].cuda()
        semantics128=self.to_tensor(semantics128)[np.newaxis,:].cuda()

        #Pass image with 128 size into model
        edges128 = self.edge_model(grays128, edges128, masks128).detach()
        outputs1 = self.inpaint_model(img128, edges128, masks128, semantics128)

        #Processing 256 size
        img256 = imresize(img, [256, 256])
        edges256 = imresize(edges, [256, 256])
        masks256 = imresize(masks, [256, 256])
        grays256 = imresize(gray, [256, 256])
        masks256 = (masks256 > 0).astype(np.uint8) * 255
        semantics256 = imresize(semantics, [256, 256])

        #Convert 256 size numpy array back to tensor
        grays256=self.to_tensor(grays256)[np.newaxis,:].cuda()
        edges256=self.to_tensor(edges256)[np.newaxis,:].cuda()
        img256=self.to_tensor(img256.astype(np.uint8))[np.newaxis,:].cuda()
        masks256=self.to_tensor(masks256)[np.newaxis,:].cuda()
        semantics256=self.to_tensor(semantics256)[np.newaxis,:].cuda()

        edges256 = self.edge_model(grays256, edges256, masks256).detach()
        outputs2 = self.inpaint_model(img256, edges256, masks256, semantics256)

        #Processing 512 size
        img512 = imresize(img, [512, 512])
        edges512 = imresize(edges, [512, 512])
        masks512 = imresize(masks, [512, 512])
        grays512 = imresize(gray, [512, 512])
        masks512 = (masks512 > 0).astype(np.uint8) * 255
        semantics512 = imresize(semantics, [512, 512])

        #Convert 512 size numpy array back to tensor
        grays512=self.to_tensor(grays512)[np.newaxis,:].cuda()
        edges512=self.to_tensor(edges512)[np.newaxis,:].cuda()
        img512=self.to_tensor(img512.astype(np.uint8))[np.newaxis,:].cuda()
        masks512=self.to_tensor(masks512)[np.newaxis,:].cuda()
        semantics512=self.to_tensor(semantics512)[np.newaxis,:].cuda()

        edges512 = self.edge_model(grays512, edges512, masks512).detach()
        outputs3 = self.inpaint_model(img512, edges512, masks512, semantics512)

        #Convert all outputs back to numpy
        outputs1=self.postprocess(outputs1)[0].cpu().numpy()
        outputs2=self.postprocess(outputs2)[0].cpu().numpy()
        outputs3=self.postprocess(outputs3)[0].cpu().numpy()

        #Resize all image back to their original size
        outputs1_origin = imresize(outputs1, [imgh, imgw])
        outputs2_origin = imresize(outputs2, [imgh, imgw])
        outputs3_origin = imresize(outputs3, [imgh, imgw])
        ##Composed all images with different weights
        
        outputs = outputs1_origin * 0.5 + outputs2_origin * 0.3 + outputs3_origin * 0.2
        #Convert it back to tensor
        masks=gray2rgb(masks)
        outputs_merged = (outputs * masks) + (img * (1 - masks))
        outputs_merged=self.to_tensor(outputs_merged.astype(np.uint8))[np.newaxis,:].cuda()
        return outputs_merged


    #Additional function
    def detectBox(self,mask):
        #Convert mask from tensor to numpy
        mask=self.postprocess(mask)[0].cpu().numpy()[:,:,0]
        # input must be two dimensional array(binary numpy)
        x,y=np.where(mask==255)# Sometime the value could be 255. 
        #find four points
        min_x,min_y=np.min(x),np.min(y)
        max_x,max_y=np.max(x),np.max(y)
        #Assume that it is a box mask, then we have their length
        mask_h=max_x-min_x+1
        mask_w=max_y-min_y+1
        #the area of box mask will be
        assume_area=mask_h*mask_w
        #If it is box mask, the number of white pixels must equal to the area
        return x.shape==assume_area

    def ensemble(self,picture,picture_mask,picture_edges,picture_gray,picture_semantics):
        #Convert image from tensor into numpy
        gt=picture
        gt_mask=picture_mask
        picture=self.postprocess(picture)[0].cpu().numpy()
        picture_mask=picture_mask[0,0,:,:].cpu().numpy()
        picture_edges=picture_edges[0,0,:,:].cpu().numpy()
        picture_gray=picture_gray[0,0,:,:].cpu().numpy()
        picture_semantics=picture_semantics[0,0,:,:].cpu().numpy()
        #Convert outpus back to tensor
        outputs=self.network(picture,picture_mask,picture_edges,picture_gray,picture_semantics)
        return self.to_tensor(outputs.astype(np.uint8))[np.newaxis,:].cuda()

# #Patch cutting part

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def network(self,patch,patch_mask,edge_patch,gray_patch,semantics_patch):
        h,w,_=patch.shape
        h1,w1=0,0
        if h%4!=0 or w%4!=0:
            h1=math.ceil(h/4)*4-h
            w1=math.ceil(w/4)*4-w
        patch=np.pad(patch,((h1,0),(w1,0),(0,0)),"constant")
        patch_mask=np.pad(patch_mask,((h1,0),(w1,0)),"constant")
        edge_patch=np.pad(edge_patch,((h1,0),(w1,0)),"constant")
        gray_patch=np.pad(gray_patch,((h1,0),(w1,0)),"constant")
        semantics_patch=np.pad(semantics_patch,((h1,0),(w1,0)),"constant")
        patch_mask = (patch_mask > 0).astype(np.uint8) * 255

        #To be notified: all the input must be in numpy form
        #Convert all element into tensor
        gray_patch=self.to_tensor(gray_patch)[np.newaxis,:].cuda()
        edges=self.to_tensor(edge_patch)[np.newaxis,:].cuda()
        patch=self.to_tensor(patch.astype(np.uint8))[np.newaxis,:].cuda()
        patch_mask=self.to_tensor(patch_mask)[np.newaxis,:].cuda()
        semantics_patch=self.to_tensor(semantics_patch)[np.newaxis,:].cuda()
        
        #place the patch into network
        # print("gray_patch: ",gray_patch.shape)
        # print("edges: ",edges.shape)
        # print("patch_mask: ",patch_mask.shape)
        edges= self.edge_model(gray_patch, edges, patch_mask).detach()
        # print("patch: ",patch.shape)
        # print("edges: ",edges.shape)
        # print("patch_mask: ",patch_mask.shape)
        output = self.inpaint_model(patch, edges, patch_mask,  semantics_patch)

        #Convert output from tensor into numpy
        output=self.postprocess(output)[0].cpu().numpy()
        if h1!=0:
            hc=[]
            for i in range(h1):
                hc.append(i)
            output=np.delete(output,hc,axis=0)
        if w1!=0:
            wc=[]
            for i in range(w1):
                wc.append(i)
            output=np.delete(output,wc,axis=1)
        return output

    def crop_to_merge(self, img, edges, masks,gray,semantic):
        #Convert all the tensor back to numpy
        img=self.postprocess(img)[0].cpu().numpy()
        masks=masks[0,0,:,:].cpu().numpy()
        edges=edges[0,0,:,:].cpu().numpy()
        gray=gray[0,0,:,:].cpu().numpy()
        semantic=semantic[0,0,:,:].cpu().numpy()

        #Cutting patchs for img
        crop_w_coords_img, crop_h_coords_img, CropPatch_img = cropSingle(img)
        #Cutting patchs for masks
        crop_w_coords_masks, crop_h_coords_masks, CropPatch_masks = cropSingle(masks)
        #Cutting patchs for edges
        crop_w_coords_edges, crop_h_coords_edges, CropPatch_edges = cropSingle(edges)
        #Cutting patchs for gray
        crop_w_coords_gray, crop_h_coords_gray, CropPatch_gray = cropSingle(gray)
        #cutting patchs for semantic
        crop_w_coords_semantic, crop_h_coords_semantic, CropPatch_semantic = cropSingle(semantic)

        for i in range(len(CropPatch_img)):
            CropPatch_img[i]=self.network(CropPatch_img[i],CropPatch_masks[i],CropPatch_edges[i],CropPatch_gray[i],CropPatch_semantic[i])#patch,patch_mask,edge_patch,gray_patch
        output=concatPatch(crop_w_coords_img, crop_h_coords_img, CropPatch_img)
        outputs_merged=self.to_tensor(output.astype(np.uint8))[np.newaxis,:].cuda()
        return outputs_merged

# when oversize happened, the 
def cropSingle(img, size=512,overlap=50):
    """
    input: a single image, which exceeds 2020Ti's processing size 
    """
    # fetch the parameters of crop
    # assumption: the stirde in width and height are the same 
    stride = size - overlap

    H, W = img.shape[:2]

    print("Parameters: original size:{}\t{}\nCrop size:{}\nCrop overlap:{}".format(H,W,size,overlap))

    CropPatch = []
    
    # by the cropsize and overlap, all the croping startpoints' coordinates can be calculated
    crop_w_coords = []
    crop_h_coords = []
    
    i = 0
    while (i + size) < W:
        crop_w_coords.append(i)
        i = i + stride
    crop_w_coords.append((W - size))

    j = 0
    while (j + size) < H:
        crop_h_coords.append(j)
        j = j + stride
    crop_h_coords.append((H - size))

    print("find the croping coordinates:{}\t{}".format(crop_w_coords, crop_h_coords))

    #return crop_w_coords, crop_h_coords
    # begin crop process
    for (i, w) in enumerate(crop_w_coords):
        for (j, h) in enumerate(crop_h_coords):
            crop = img[h:h+size, w:w+size]
            CropPatch.append(crop)
            #cv.imwrite(str(i)+'_'+str(j)+'.jpg', crop)
    return crop_w_coords, crop_h_coords, CropPatch

def visualizeCrop(imgs, w_coords, h_coords, size=512):
    # use this function to see the croping plan for a oversize image
    def getColor():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return (r,g,b)

    for w in w_coords:
        for h in h_coords:
            cv.rectangle(imgs, (w,h), (w+size,h+size), getColor(), 8)

    cv.imwrite("./crop_result.jpg", imgs)

def concatPatch(crop_w_coords, crop_h_coords, CropPatch, size=512):
    """
    input: the crop patch of a oversized image
    output: the whole image after concatenation
    """
    # the concat of two patch is divided into three part:
    # each of independent part (2) + overlap part (1)
    # there is a weighted addition to obtain the overlap part

    # first, get the column part of image
    w_num = len(crop_w_coords)
    h_num = len(crop_h_coords)

    PatchwithColumn = []
    Columns = []
    # each element in the PatchwithColumn is a list of crops that can concat to a column of original image
    # Column is a list of several column of the original image
    def get_overlap(coords, size):
        overlaps = []
        for (idx, coord) in enumerate(coords):
            if (idx+1) == len(coords):
                break
            else:
                overlaps.append(coord + size - coords[idx+1])
        return overlaps

    def concat_to_Column(crop_patch, coords, size):
        """
        crop_patch: a list of crop which can concat to a column of original image
        the process can be interpreted as calculate the UNION SET 
        """
        overlaps = get_overlap(coords, size)
        column_final = crop_patch[0]
        for i in range(len(overlaps)):
            h1 = column_final.shape[0]
            h2 = crop_patch[i+1].shape[0]
            this_overlap = overlaps[i]
            a1 = 0 # start point of upper
            a2 = h1 - this_overlap # independent part of concat element 1
            b1 = 0
            b2 = this_overlap
            sub_union_1 = column_final[a1:a2, :]
            intersection_1 = column_final[a2:h1, :]
            intersection_2 = crop_patch[i+1][b1:b2, :]
            sub_union_2 = crop_patch[i+1][b2:h2, :]
            intersection = 0.5 * intersection_1 + 0.5 * intersection_2
            column_temp = np.concatenate((sub_union_1, intersection, sub_union_2), axis=0)
            column_final = column_temp

        return column_final

    def concat_to_Row(crop_patch, coords, size):
        """
        column_patch: a list of column crop which can concat to the entire original image
        the process can be interpreted as calculate the UNION SET 
        """
        overlaps = get_overlap(coords, size)
        row_final = crop_patch[0]
        for i in range(len(overlaps)):
            w1 = row_final.shape[1]
            w2 = crop_patch[i+1].shape[1]
            this_overlap = overlaps[i]
            a1 = 0 # start point of upper
            a2 = w1 - this_overlap # independent part of concat element 1
            b1 = 0
            b2 = this_overlap
            sub_union_1 = row_final[:, a1:a2]
            intersection_1 = row_final[:, a2:w1]
            intersection_2 = crop_patch[i+1][:, b1:b2]
            sub_union_2 = crop_patch[i+1][:, b2:w2]
            intersection = 0.5 * intersection_1 + 0.5 * intersection_2
            row_temp = np.concatenate((sub_union_1, intersection, sub_union_2), axis=1)
            row_final = row_temp

        return row_final

    for i in range(0, len(CropPatch), h_num):
        column_single = CropPatch[i:i+h_num]
        PatchwithColumn.append(column_single)

    for patchwithcolumn in PatchwithColumn:
        _column = concat_to_Column(patchwithcolumn, crop_h_coords, size)
        Columns.append(_column)

    concat_final = concat_to_Row(Columns, crop_w_coords, size)
    return concat_final
    # cv.imwrite("./reconstruction.jpg", concat_final)
