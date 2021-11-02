#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hseghetto/PICME-Deep-Learning/blob/main/coditional_grow_gan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[6]:


import tensorflow as tf

import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import os

#from IPython import display

path = "scratch/08429/hss1999/"
#path = ""

# In[2]:


#tf.__version__
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],True)


# #Loading dataset
# 

# In[3]:


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

"""(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')"""

train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

print(train_images.shape)


# In[4]:


upsample = tf.keras.layers.UpSampling2D(size=(2,2))
downsample = tf.keras.layers.AveragePooling2D(pool_size=(2,2))

shape = upsample(train_images[0:10]).shape
print(shape)

shape = downsample(train_images[0:10]).shape
print(shape)

assert shape[-3]%2 == 0 and shape[-2]%2 == 0

WIDTH = shape[-3]//2 #Smallest width to be used in generator, equivalente to 1/4 of the original image size
HEIGHT = shape[-2]//2
CHANNELS = shape[-1]
LABEL_DIM = len(np.unique(train_labels))
print(WIDTH,HEIGHT,CHANNELS,LABEL_DIM)


# In[5]:


BUFFER_SIZE = 60000
BATCH_SIZE = 64
noise_dim = 100

# Batch and shuffle the data
train_dataset14 = tf.data.Dataset.from_tensor_slices((downsample(train_images),train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset28 = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset56 = tf.data.Dataset.from_tensor_slices((upsample(train_images),train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# In[6]:


class WeightedSum(tf.keras.layers.Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

class LeakyConv2D(layers.Layer):
    def __init__(self, channels, kernel=(5,5), strides=(1,1)):
        super(LeakyConv2D, self).__init__()
        self.Conv = layers.Conv2D(channels, kernel, strides, padding ='same', use_bias=False)
        self.Norm = layers.BatchNormalization()
        self.ReLU = layers.LeakyReLU()

    def call(self, inputs):
        x = self.Conv(inputs)
        x = self.Norm(x)
        out = self.ReLU(x)

        return out

class LeakyConv2DTranspose(layers.Layer):
    def __init__(self, channels, kernel=(5,5), strides=(1,1)):
        super(LeakyConv2DTranspose, self).__init__()
        self.Conv = layers.Conv2DTranspose(channels, kernel, strides, padding ='same', use_bias=False)
        self.Norm = layers.BatchNormalization()
        self.ReLU = layers.LeakyReLU()

    def call(self, inputs):
        x = self.Conv(inputs)
        x = self.Norm(x)
        out = self.ReLU(x)

        return out

class LabelEmbedding(layers.Layer):
    def __init__(self, out_shape, label_dim=10, embed_size=50, label_size=1):
        super(LabelEmbedding, self).__init__()
        self.Embedding = layers.Embedding(label_dim, embed_size ,input_length=label_size) 
        self.Flatten = layers.Flatten()
        self.Dense = layers.Dense(np.prod(out_shape))
        self.Reshape = layers.Reshape(out_shape)
        
    def call(self, inputs):

        x = self.Embedding(inputs)
        x = self.Flatten(x)
        x = self.Dense(x)
        out = self.Reshape(x)

        return out



# #Model

# ##Generator

# In[7]:


def generator_model():
    
    input_shape=(noise_dim,)
    input1 =  layers.Input(input_shape)

    x = layers.Dense(WIDTH*HEIGHT*32, use_bias = False)(input1)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    label = layers.Input(1,)
    y = LabelEmbedding((WIDTH*HEIGHT*32,),LABEL_DIM)(label)

    x = layers.Concatenate()([x,y])

    x = layers.Reshape((WIDTH,HEIGHT,64))(x)
 
    x = LeakyConv2DTranspose(64, strides = (2,2))(x)

    x = LeakyConv2DTranspose(64)(x)

    x = layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(  1, 1), padding='same', use_bias=False)(x)
    
    model = tf.keras.Model([input1,label],x)
    print("Generator model created succesfully")
    return model


# In[8]:


def generator_expand(old_gen):
    input = old_gen.inputs
    x1 = old_gen.layers[-1].output
    x1 = upsample(x1)

    x2 = old_gen.layers[-1].input
    
    x2 = LeakyConv2DTranspose(64, strides = (2,2))(x2)
    x2 = LeakyConv2DTranspose(64)(x2)

    x2 = layers.Conv2DTranspose(CHANNELS, (5, 5), strides=(  1, 1), padding='same', use_bias=False)(x2)

    out = WeightedSum()([x1,x2])
    
    model_weighted = tf.keras.Model(input,out)
    model = tf.keras.Model(input,x2)
    print("Generator model expanded succesfully")
    return model_weighted, model


# In[9]:


# gen = generator_model()
# gen.summary()
# tf.keras.utils.plot_model(gen,to_file="gen.png")

# new_gen,_ = generator_expand(gen)
# new_gen.summary()
# tf.keras.utils.plot_model(new_gen,to_file="gen_exp.png")


# ##Discriminator

# In[10]:


def discriminator_model():
    input_shape = (WIDTH*2,HEIGHT*2,CHANNELS)
    input1 = layers.Input(input_shape)

    x = LeakyConv2D(1)(input1)

    label = layers.Input(1)

    y = LabelEmbedding((WIDTH*2,HEIGHT*2,CHANNELS,),LABEL_DIM)(label)

    x = layers.Concatenate()([x,y])

    x = LeakyConv2D(64)(x)
    x = LeakyConv2D(64,strides = (2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    model = tf.keras.Model((input1,label),x)
    print("Discriminator model created succesfully")
    return model


# In[11]:


def discriminator_expand(old_disc):
    size = old_disc.inputs[0].shape[1]*2
    input_shape = (size,size,CHANNELS)

    input1 = layers.Input(input_shape)
    x1 = LeakyConv2D(1)(input1)

    label = layers.Input(1)
    y1 = LabelEmbedding((size,size,CHANNELS,),LABEL_DIM)(label)

    x1 = layers.Concatenate()([x1,y1])
    x1 = LeakyConv2D(64)(x1)
    x1 = LeakyConv2D(64, strides=(2,2))(x1)

    x2 = downsample(input1)
    x2 = old_disc.layers[2](x2)

    y2 = old_disc.layers[3](label)
    
    x2 = old_disc.layers[4]((x2,y2))
    x2 = old_disc.layers[5](x2)

    x2 = WeightedSum()((x2,x1))

    for old_layer in old_disc.layers[6:]:
        x1 = old_layer(x1)
        x2 = old_layer(x2)

    model_weighted = tf.keras.Model([input1,label],x2)
    model = tf.keras.Model([input1,label],x1)
    print("Discriminator model expanded succesfully")
    return model_weighted, model


# In[12]:


#disc = discriminator_model()

#disc.summary()
#tf.keras.utils.plot_model(disc,to_file="disc.png")

#new_disc, disc = discriminator_expand(disc)

#new_disc.summary()
#tf.keras.utils.plot_model(disc,to_file="disc_exp.png")


# #Optimizer & Losses

# In[13]:


#@tf.function
def generator_loss(fake_output):
    return tf.reduce_sum(fake_output)


# In[14]:


#@tf.function
def discriminator_loss(real_output, fake_output, penalty = 0):
    return tf.reduce_sum(real_output - fake_output) + penalty


# In[15]:


c = 0.1
generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
#discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-4,clipnorm = c) #using clipnorm to satisfy WGAN requirements
discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-4)

#generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-08)


# #Training Loop

# In[16]:


noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
noise = tf.random.normal([num_examples_to_generate, noise_dim])
labels = np.array(range(num_examples_to_generate))%LABEL_DIM
labels = tf.convert_to_tensor(labels,dtype=noise.dtype)

seed = (noise, labels)


# In[17]:


#@tf.function
def train_step(images,labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator((noise,labels), training=True)

        real_output = discriminator((images,labels), training=True)
        fake_output = discriminator((generated_images,labels), training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[18]:


#@tf.function #cant use same tf function with different model shapes
def train_step_gp(images,labels):
    noise = tf.random.normal([images.shape[0], noise_dim])
    e = tf.random.uniform([images.shape[0],1,1,1],minval = 0,maxval = 1)

    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as penalty_tape:
            generated_images = generator((noise,labels), training=True)

            weighted_images = e*images + (1-e)*generated_images

            weighted_output = discriminator((weighted_images,labels), training = True)

        real_output = discriminator((images,labels), training = True)
        fake_output = discriminator((generated_images,labels), training = True)

        penalty = penalty_tape.gradient(weighted_output,weighted_images) #gradient
        penalty = tf.sqrt(tf.reduce_sum(tf.square(penalty), axis=[1, 2, 3])) #norm
        penalty = 100*tf.reduce_mean(tf.math.square(penalty-1)) # k*(norm -1)^2

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, penalty)


    gradients_of_generator = tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# In[19]:


def train(dataset,Epochs):

    print("Starting training")
    for epoch in range(Epochs):
        start = time.time()

        #print("Epoch {}".format(epoch+1))

        for image_batch, label_batch in dataset:
            #print(".",end="")
            train_step_gp(image_batch,label_batch)

        # Produce images for the GIF as you go
        #display.clear_output(wait=True)
        generate_and_save_images(epoch + 1,seed)
        
        print ("Time for epoch {}/{} is {} sec".format(epoch + 1,Epochs, time.time()-start))


# In[20]:


def train_fade_in(dataset,Epochs):

    print("Starting training with fade-in")
    for epoch in range(Epochs):
        start = time.time()
        #print("Epoch {}".format(epoch+1))

        alpha = (epoch+1)/Epochs #0 to 1
        for image_batch, label_batch in dataset:
            #print(".",end="")
            for layer in discriminator.layers:
                if isinstance(layer,WeightedSum):
                    tf.keras.backend.set_value(layer.alpha,alpha)
            for layer in generator.layers:
                if isinstance(layer,WeightedSum):
                    tf.keras.backend.set_value(layer.alpha,alpha)
            
            train_step_gp(image_batch,label_batch)


        # Produce images for the GIF as you go
        #display.clear_output(wait=True)
        generate_and_save_images(epoch + 1,seed)
        
        print ("Time for epoch {}/{} is {} sec".format(epoch + 1,Epochs, time.time()-start))


# In[21]:


def generate_and_save_images(epoch, test_input):
    predictions = generator(test_input, training = False)
    (examples, width, height, channels) = predictions.shape 
    
    mode = 'RGB'
    if channels==1:
        mode = 'L'
        predictions = np.squeeze(predictions)
        
    result = PIL.Image.new(mode,(4*width,4*height))
    
    for i in range(min(16,examples)):
        image = np.array((predictions[i]*127.5+127.5)).clip(0,255).squeeze().astype("uint8")
        image = PIL.Image.fromarray(image,mode)
        result.paste(im=image,box=(width*(i//4),height*(i%4)))
    
    result.save(path+'cg_gan/nn_{}_step_{}_epoch_{:04d}.png'.format(NN,TRAIN,epoch),"PNG")
    plt.imshow(np.asarray(result),cmap="gray")
    plt.show()


# ### Testing

# In[22]:


#get_ipython().run_cell_magic('time', '', 'for i in range(10):\n    image = tf.random.normal([1,2,2,1])\n    upsize = upsample(image)\nprint(upsize.shape)\nprint(type(upsize))\nprint(type(image))\n\n(width,height,channels) = train_images[0].shape\nmode = \'RGB\'\nif channels==1:\n    mode = \'L\'\nresult = PIL.Image.new(mode,(3*width,3*height))\n\nfor i in range(3):\n    image = downsample(upsample(train_images[i:i+1]))\n    image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")\n    image = PIL.Image.fromarray(image,mode)\n    result.paste(im=image,box=(width*0,height*i))\n    \n    image = train_images[i:i+1]\n    image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")\n    image = PIL.Image.fromarray(image,mode)\n    result.paste(im=image,box=(width*1,height*i))\n    \n    image = upsample(downsample(train_images[i:i+1]))\n    image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")\n    image = PIL.Image.fromarray(image,mode)\n    result.paste(im=image,box=(width*2,height*i))\n\nplt.imshow(np.asarray(result),cmap="gray")\nplt.axis("off")\nplt.show()')


# In[23]:


(width,height,channels) = train_images[0].shape
mode = 'RGB'
if channels==1:
    mode = 'L'
result = PIL.Image.new(mode,(int(3.5*width),2*height))

for i in range(4):
    image = downsample(train_images[i:i+1])
    image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")
    image = PIL.Image.fromarray(image,mode)
    result.paste(im=image,box=(width*0,int(height*i/2)))
    
for i in range(2):
    image = train_images[i:i+1]
    image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")
    image = PIL.Image.fromarray(image,mode)
    result.paste(im=image,box=(int(width/2),height*i))

i=0
image = upsample(train_images[i:i+1])
image = np.array((image*127.5+127.5)).clip(0,255).squeeze().astype("uint8")
image = PIL.Image.fromarray(image,mode)
result.paste(im=image,box=(int(width*3/2),0))

plt.imshow(np.asarray(result),cmap="gray")
plt.axis("off")
plt.show()


# #Model training

# In[24]:


try:
    NN = NN + 1
except:
    og_train_step = train_step_gp
    NN = 1
         


# In[9]:


try:
    os.mkdir(path+"cg_gan")
except:
    pass


# In[ ]:


#OG model
TRAIN = 0
train_step_gp = tf.function()(og_train_step)

generator = generator_model()
discriminator = discriminator_model()

train(train_dataset14,20)
TRAIN = TRAIN+1

#1nd expansion
generator,g = generator_expand(generator)
discriminator,d = discriminator_expand(discriminator)

train_step_gp = tf.function()(og_train_step)

train_fade_in(train_dataset28,10)
TRAIN = TRAIN+1

train(train_dataset28,20)
TRAIN = TRAIN+1

#2nd expansion
generator,g = generator_expand(g)
discriminator,d = discriminator_expand(d)

train_step_gp = tf.function()(og_train_step)

train_fade_in(train_dataset56,10)
TRAIN = TRAIN+1

train(train_dataset56,20)
TRAIN = TRAIN+1


# In[ ]:


for i in range(10):
    seed = (noise,tf.ones(num_examples_to_generate)*i)

    predictions = generator(seed, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()


# In[4]:


fp_in = path+"cg_gan/nn_{}*.png".format(NN)
fp_out = path+"cg_gan/nn_{}_scaled.gif".format(NN)

img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))]

img = img.resize(imgs[-1].size, PIL.Image.ANTIALIAS)
for i in range(len(imgs)):
    imgs[i] = imgs[i].resize(imgs[-1].size, PIL.Image.ANTIALIAS)

img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0)


# In[ ]:


fp_in = path+"cg_gan/nn_{}*.png".format(NN)
fp_out = path+"cg_gan/nn_{}_real.gif".format(NN)

imgs = [PIL.Image.open(f) for f in sorted(glob.glob(fp_in))]

result = PIL.Image.new("RGB", imgs[-1].size, (0,0,0))
for i in range(len(imgs)):
    x = int(result.size[0]/2-imgs[i].size[0]/2)
    y = int(result.size[1]/2-imgs[i].size[1]/2)
    
    aux = imgs[i].copy()
    imgs[i] = result.copy()
    imgs[i].paste(aux,box=(x,y))

result.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0)


# In[ ]:




