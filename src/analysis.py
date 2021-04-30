#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt

ITEMS = ['canary', 'robin', 'daisy', 'rose', 'oak', 'pine','salmon', 'sunfish']

def get_reps(path,
             exp = 'semc-simple-exp1-lr1e-4', 
             items = ['canary','robin','daisy','rose','pine','oak','salmon','sunfish'],
             epoch_params = (50,400,150)):
  
    """  
    Input
        path : file path to saved .pt files
        exp : label for experiment
        items : list of item names
        epoch_params : (start, stop, step) specify epochs to extract reps from 

    Output
        rep_dict : dictionary to store representations
        -- keys: items 
        -- values: array of representations of size [num_epochs x rep_size] 
        -- e.g. rep_dict['canary'] = [[rep0], [rep1], [rep2]]  
    """

    start, stop, step = epoch_params
    model = 'simple cnn' if exp.find('-simple-') > -1 else 'resnet'

    # Create dict to store representations for each item
    rep_dict = {x: [] for x in items}

    # Loop through files for each item and epochs
    for i in items:
        for ep in np.arange(start,stop,step):
            filename = model+'/'+exp+'/version_0/'+'epoch_'+str(ep)+'/'+i+'.pt'
            images = torch.load(path+'/'+filename, map_location=torch.device('cpu'))
            images_array = images.detach().numpy()

            # [32 x 29 x 29] -> [26912 x 1]
            flattened_rep = images_array.flatten()

            # Append to dictionary: item i's representation at epoch ep
            rep_dict[i].append(flattened_rep)
                
    return rep_dict

def get_reps_from_dict(rep_dict,items=ITEMS):
    """  
    Assumes there are only 3 saved representations: (rep0, rep1, rep2)

    Input
        rep_dict : dictionary to store representations
        -- keys: items 
        -- values: array of representations of size [num_epochs x rep_size] 
        -- e.g. rep_dict['canary'] = [[rep0], [rep1], [rep2]]  
        items : list of item names

    Output  
        rep0, rep1, rep2: three numpy arrays of size [num_items x rep_size]
        -- representation layers at successive points in training 
        -- (e.g. epoch = 100, 200, 500)
    """  
    rep0 = np.array([rep_dict[i][0] for i in items])
    rep1 = np.array([rep_dict[i][1] for i in items])
    rep2 = np.array([rep_dict[i][2] for i in items])
    return rep0, rep1, rep2

# Modified code from HW
def plot_rep(rep1,rep2,rep3,items,epoch_params=(50,400,150)):
    """  
    Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    using bar graphs

    Input
    Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    items : [nitem list] of item names
    epoch_params : (start, stop, step) specify epochs to extract reps from 
    """
    start, stop, step = epoch_params
    nepochs_list = list(np.arange(start, stop, step))
    nrows = len(items)
    R = np.dstack((rep1,rep2,rep3))
    mx = R.max()
    mn = R.min()
    depth = R.shape[2]
    count = 1
    plt.figure(1,figsize=(8.4,16.8))
    for i in range(nrows):
        for d in range(R.shape[2]):
            plt.subplot(nrows, depth, count)
            rep = R[i,:,d]
            plt.bar(range(rep.size),rep)
            plt.ylim([mn,mx])
            plt.xticks([])
            plt.yticks([])
            if d==0:
                plt.ylabel(items[i])
            if i==0:
                plt.title("epoch " + str(nepochs_list[d]))
            count += 1
    plt.savefig('rep.png')
    plt.show()

# Modified code from HW
def plot_dendo(rep1,rep2,rep3,items,epoch_params=(50,400,150)):
    """  
    Compares Representation Layer activations of Items at three different times points in learning (rep1, rep2, rep3)
    using hierarchical clustering

    Input
    Each rep1, rep2, rep3 is a [nitem x rep_size numpy array]
    items : [nitem list] of item names
    epoch_params : (start, stop, step) specify epochs to extract reps from 
    """

    from scipy.cluster.hierarchy import dendrogram, linkage

    print('Representation size:', rep1.shape)

    start, stop, step = epoch_params
    nepochs_list = list(np.arange(start, stop, step))
    linked1 = linkage(rep1,'single')
    linked2 = linkage(rep2,'single')
    linked3 = linkage(rep3,'single')
    mx = np.dstack((linked1[:,2],linked2[:,2],linked3[:,2])).max()+0.1
    plt.figure(2,figsize=(7,12))
    plt.subplot(3,1,1)
    dendrogram(linked1, labels=items, color_threshold=0)
    plt.ylim([0,mx])
    plt.title('Hierarchical clustering; ' + "epoch " + str(nepochs_list[0]))
    plt.ylabel('Euclidean distance')
    plt.subplot(3,1,2)
    plt.title("epoch " + str(nepochs_list[1]))
    dendrogram(linked2, labels=items, color_threshold=0)
    plt.ylim([0,mx])
    plt.subplot(3,1,3)
    plt.title("epoch " + str(nepochs_list[2]))
    dendrogram(linked3, labels=items, color_threshold=0)
    plt.ylim([0,mx])
    plt.savefig('dendo.png')

if __name__ == "__main__":

    path = '/Users/francescaguiso/semantic_cognition/data/model artifacts'
    # path = '/content/drive/MyDrive/model artifacts'

    model_types = {'simple cnn','resnet'}

    # Plot feature maps
    resnet_experiments = ['semc-resnet-exp0-lr1e-4', 'semc-resnet-exp2-lr1e-6']
    simple_cnn_experiments = ['semc-simple-exp1-lr1e-4', 'semc-simple-exp3-lr1e-6']

    experiments = {'simple cnn': simple_cnn_experiments,
                'resnet': resnet_experiments}

    model = 'simple cnn'
    for exp in experiments[model]:
        for i in ITEMS:
            for ep in np.arange(50,400,150):
                filename = model+'/'+exp+'/version_0/'+'epoch_'+str(ep)+'/'+i+'.pt'
                images = torch.load(path+'/'+filename,map_location=torch.device('cpu'))
                images_array = images.detach().numpy()

            f = 0
            # Plot individual feature maps:
            for img in images_array: # images_array shape: [32 x 29 x 29]
                plt.imshow(img,cmap='gray')
                plt.title(f"Item: {i} (Epoch: {ep}, Exp: {exp})")
                plt.savefig(path+'/img/'+i+'/'+'item_'+str(f)+'_'+exp+'_'+str(ep))
                #plt.show()
                f+=1

    cnn_rep_dict = get_reps(path=path, exp = 'semc-simple-exp1-lr1e-4')
    cnn_rep0, cnn_rep1, cnn_rep2 = get_reps_from_dict(cnn_rep_dict,items=ITEMS)

    plot_rep(cnn_rep0, cnn_rep1, cnn_rep2, items=ITEMS)

    plot_dendo(cnn_rep0,cnn_rep1,cnn_rep2, items=ITEMS)

    # TODO: Analysis using Resnet representations: [2048 x 1]