# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:04:42 2019
@author: marco
"""
def bd(path,optimizer='SGD',loss='categorical_crossentropy',metrics=["accuracy"],*args):
    def glob(path):
        from glob import glob
        any = glob(path)
        return any
    def shaping(files):#get files
        from cv2 import imread
        from numpy import array
        imgs,av = [],[]
        for path in files:
            p = path_name+path[-1]+'/*.jpg'
            p = glob(p)
            for f in p:
                av.append(int(path[-1]))
                image = imread(f) 
                imgs.append(image) 
        av = array(av,dtype='uint8')
        imgs = array(imgs)
        return imgs,av

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
    from tensorflow.keras.utils import np_utils
    #setting types
    files = glob(path+'*')
    x_train, y_train = shaping(files)
    f = [len(x_train[0]),len(x_train[0,0]),len(x_train[0,0,0])]
    x_train = x_train.astype('float64')/255
    y_train = np_utils.to_categorical(y_train, len(files)).astype('uint8')
    #test unit(validation)
    x_test = x_train
    y_test = y_train
    #pr-process
    model = Sequential()
    model.add(Conv2D(16, (4,3), input_shape=(f[0],f[1],f[2])))
    model.add(Conv2D(16, (4,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (4,3)))
    model.add(Conv2D(16, (4,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #dense
    model.add(Dense(units=32,activation="tanh",name="dense_1"))
    model.add(Dense(units=16,activation="tanh",name="dense_2"))
    model.add(Dense(units=8,activation="tanh",name="dense_3"))
    model.add(Dense(units=len(files),activation= 'softmax' if len(files)>2 else 'sigmoid',name="dense_f"))
    try:
        model.load_weights("my_weight.h5")
    except OSError:
        print("weights do not exist")
    #compile
    model.compile(optimizer=optimizer, 
                  loss=loss,
                  metrics=metrics)
    return model,x_test,x_train,y_test,y_train

##---------------------------------------------------------------------
#path_name
path_name = "c:/Users/marco/Downloads/drive/"

model,x_test,x_train,y_test,y_train = bd(path_name)
#train
history = model.fit(x_train, y_train, epochs=300)
#test
resultado = model.evaluate(x_test, y_test, batch_size=2)    
#save
print(resultado)
model.save("my_model_test.h5",overwrite=True)
model.save_weights("my_weight.h5",overwrite=True)

##---------------------------------------------------------------------
