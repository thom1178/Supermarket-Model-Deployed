# Keras Methods #####
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense,GlobalAveragePooling2D
from keras.engine import InputLayer
#from tensorflow.contrib.keras.python.keras.backend import clear_session
#clear_session()
import numpy as np
import cv2

def to_fully_conv(model):
    """
    This function converts fully connencted layers into convolutional layers for localization.
    Modified from the following:
    https://stackoverflow.com/questions/41161021/how-to-convert-a-dense-layer-to-an-equivalent-convolutional-layer-in-keras
    """
    new_model = Sequential()

    #input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    #new_model.add(input_layer)

    for layer in model.layers:
        print(layer)
        if "GlobalAveragePooling2D" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
        else:
            new_layer = layer
            
        new_model.add(new_layer)
        flattened_ipt = False
    return new_model
def process_pred_img(img, w = 224, h = 224):
    """
    This function resizes and reshapes an input image for processing
    """
    img = cv2.resize(img,(h,w))
    img = img.reshape(1,w,h,3)
    return img

def localizee(model,unscaled, W = 300, H = 300, THRESHOLD = .8, EPSILON = 0.02):
    """
    This function iterates through scales for a fixed w, h image and gets a heatmap. 
    The function return a list of bounding over the scales.
    It thresholds the heatmap at .7, and appends to a list.
    This function follows from: https://github.com/lars76/object-localization/blob/master/example_3/test.py
    """
    scales = np.power(0.8, np.arange(0, 5)) #[0.3, 0.4,..., 0.9, 1.0]
    list_of_heatmaps = []

    
    
    IMAGE_SIZE = 224
    
    bounding_boxes = [] #return list of bounding boxes w/ corrosponding scale
    if unscaled is None:
        #No image found
        print("No such image")
        return
    image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE)) #(300,300)
    for i,scale in enumerate(scales[::-1]):
        #Scale the image
        image_copy = image.copy() 
        unscaled_copy = unscaled.copy()
        feat_scaled = process_pred_img(image_copy, w = int(W*scale), h= int(H*scale) )

        region = np.squeeze(model.predict(feat_scaled))
        output = np.zeros(region[:,:,0].shape, dtype=np.uint8)
        
        output[region[:,:,0] > THRESHOLD] = 1 
        output[region[:,:,0] <= THRESHOLD] = 0
        major = cv2.__version__.split('.')[0]
        if major == '3':
            _,contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, EPSILON * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            try:
                #Sometimes output is a scalar and has no shape
                x0 = np.rint(x * unscaled.shape[1] / output.shape[1]).astype(int)
                x1 = np.rint((x + w) * unscaled.shape[1] / output.shape[1]).astype(int)
                y0 = np.rint(y * unscaled.shape[0] / output.shape[0]).astype(int)
                y1 = np.rint((y + h) * unscaled.shape[0] / output.shape[0]).astype(int)
            except Exception as e:
                continue
            
            bounding_boxes.append((x0, y0, x1, y1))
            cv2.rectangle(unscaled_copy, (x0, y0), (x1, y1), (255, 0, 255), 10)
        #cv2.imwrite("localized_sample/localized_INDEX_" + str(i) + "_FILE_" + filepath.split("/")[-1], unscaled_copy)

    return np.array(bounding_boxes)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes	
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap >= overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



