import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf   


choice=int(input('Enter no of layers\n'))
choice=int(choice/2)

tr_dir = '/Users/vikashkumar/Desktop/train'
test_dir = '/Users/vikashkumar/Desktop/test'
TEST='/Users/vikashkumar/Desktop/testimage'
img_size = 50
lr = 1e-3

model_name= 'dogsvscats-{}-{}.model'.format(lr, '6conv-layer-testing19') # to remember the model

def label_img(img):
    word_label = img.split('.')[-3]
  
    if word_label == 'cat': return [1,0]
   
    elif word_label == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    for img in os.listdir(tr_dir):
        label = label_img(img)
        path = os.path.join(tr_dir,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in os.listdir(TEST):
        path = os.path.join(TEST,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
    

#train_data = create_train_data()
# If  dataset already created
train_data = np.load('train_data.npy')

##CNN 
def layers(n):
    global model
    convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

    for i in range(n):
        #f=(i%2+1)
        print('layer number',i+1)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)      
    

    #fully connected layer
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5) 

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(model_name)):
        model.load(model_name)
        print('model loaded!')
    train = train_data[:-500]
    test = train_data[-500:]
    X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
    Y = [i[1] for i in train]       

    test_x = np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
    test_y = [i[1] for i in test]
    model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=model_name)     

    model.save(model_name)

layers(choice)



import matplotlib.pyplot as plt

test_data = process_test_data()

i=0
for data in test_data:
    i+=1
    # if we already have  saved:
    #test_data = np.load('test_data.npy')   

    fig=plt.figure()    
    

    img_num = data[1]  #label of image
    img_data = data[0] #array of image
    y = fig.add_subplot(3,4,1)
    orig = img_data
    data = img_data.reshape(img_size,img_size,1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1: 
        str_label='Dog'
        r=1
    else: 
        str_label='Cat'
        r=0
    print(img_num,' ',r)      
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    if i>100:
        break
plt.show()



