
########Set up model

from keras.applications import vgg16
from keras.models import Model

model = vgg16.VGG16(weights='imagenet', include_top=True)
model2 = Model(model.input, model.layers[-2].output)

######get some imagenet stimuli to play around with

#Get list of images
imgflist = r'C:\matlab_files\lineups_modelling\tool_stimuli\*.jpeg'  #wildcard string for detection
import glob     #Can use this to detect dir contents with wildcard string
imgflist = glob.glob(imgflist)
sorted(imgflist)  #now sort

import keras.utils as image
# from tensorflow.keras.utils import load_img
from IPython.display import display
import numpy as np 

from keras.applications.vgg16 import preprocess_input
from keras.models import Model

dat = []
imgs = []
labs = []
for count, imgf in enumerate(imgflist):
    
    #store labels
    if count < 10:
        labs.append(1)
    else:
        labs.append(0)  
    
    #import image
    img = image.load_img(imgf, target_size=(224,224))
    imgs.append(img)
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)

    #    The images are converted from RGB to BGR, then each color channel is 
#    zero-centered with respect to the ImageNet dataset, without scaling.
#    x continues to be 1*224*224*3
    x = preprocess_input(img_arr)
    
    #preds is 1*4096
    preds = model2.predict(x) #requires "from keras.models import Model" with capital M
    
    #So we sould end up with dat, a list with one slice / item per image
    dat.append(preds[0])
    

########t-SNE visualisation
from sklearn.manifold import TSNE

#Perplexity must be less than number of samples
tsne = TSNE(n_components=2, random_state=42,perplexity=len(dat)-1)
X_tsne = tsne.fit_transform(np.asarray(dat))    #samples (i.e. pictures) by dimensions (2)
#print(tsne.kl_divergence_)

import matplotlib.pyplot as plt

plt.scatter(X_tsne[:, 0],X_tsne[:, 1],c=labs)
plt.show()


#######PCA visualisation
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.asarray(dat))

plt.scatter(X_pca[:, 0],X_pca[:, 1],c=labs)
plt.show()


#########Euclidean distance matrix
from scipy.spatial.distance import pdist, squareform

Ematrix = squareform(pdist(np.asarray(dat),'euclidean'))

import seaborn as sns
p1 = sns.heatmap(Ematrix)


###########Nearest neighbour accuracy
for 



# import plotly.express as px
# # fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labs)
# fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labs)

# fig.update_layout(
#     title="t-SNE visualization of Custom Classification dataset",
#     xaxis_title="First t-SNE",
#     yaxis_title="Second t-SNE",
# )
# fig.show()

print('audi5000')