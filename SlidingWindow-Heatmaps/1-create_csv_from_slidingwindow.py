#!/usr/bin/env python

'''
Do heatmap from a input image
'''

import os
import sys
import tensorflow as tf
from PIL import Image


def print_outpu():
    '''
    dump the output file
    '''
    file_output = "output_class.csv"
    sys.stdout = open(file_output, 'w')
    print 'cont, human_string, score, posx, posy, new_width, new_height, divisor'

def sliding_window(imageinput):
    '''
    Do the sliding windows with the image input
    '''
    posx = 0
    posy = 0
    cont = 0
    if os.path.isfile(imageinput.file_input):
        imagen = Image.open(imageinput.file_input)
        size = imagen.size
        width = size[0]
        height = size[1]
        # read labels of model tf
        label_lines = [line.rstrip() for line in tf.gfile.GFile("data/retrained_labels.txt")]
        print 'cont, human_string, score, divisor, posx, posy, posx, new_width, \
            new_width, new_height, new_height, posy'
        # open the train model
        with tf.gfile.FastGFile("data/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
         #ratio and number of divisors
        for divisor in range(1, 2, 1):
            posx = 0
            posy = 0
            new_width = width / divisor
            new_height = height / divisor
            solape_x = new_width * imageinput.solapex
            solape_y = new_height * imageinput.solapey
            #create image directory
            new_path = 'solution/divisor_' + str(divisor)
            if not os.path.isdir(new_path):
                os.makedirs(new_path)
            while new_height <= height:
                posx = 0
                new_width = width / divisor
                while new_width <= width:
                    cont += 1
                    caja = (posx, posy, new_width, new_height)
                    #print 'posicion x %f posicion y %f divisor %i \
                    # new_width %f new_height %f solape_x %f solape_y \
                    # %f' % (posx, posy, divisor, new_width, new_height, solape_x, solape_y)
                    #cut box
                    region = imagen.crop(caja)
                    new_name = new_path + '/' + str(cont) + 'imagen_' \
                        + str(posx) + '_' + str(posy) + '.jpg'
                    region.save(new_name)
                    # open image in tf
                    image_data = tf.gfile.FastGFile(new_name, 'rb').read()
                    # compare image with model tf
                    with tf.Session() as sess:
                        # Feed the image_data as input to the graph and get first prediction
                        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                        predictions = sess.run(softmax_tensor, \
                            {'DecodeJpeg/contents:0': image_data})
                        sess.close()
                    # sort the output
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    # build the arrays
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        #print '%s (score = %.5f)' % (human_string, score)
                        print '%i,%s,%.5f,%i,POLYGON ((%i,%i),(%i,%i),(%i,%i),(%i,%i))' % (cont,\
                            human_string, score, divisor, posx, posy, posx, new_width, new_width, \
                            new_height, new_height, posy)
                    posx += solape_x
                    new_width += solape_x
                posy += solape_y
                new_height += solape_y
        imagen.close()
    else:
        print "Image not found"
        sys.exit(1)

class ImageInput(object):
    '''
    create class in file input
    '''
    def __init__(self, file_input, solapex, solapey):
        self.file_input = file_input
        self.solapex = solapex
        self.solapey = solapey

def main():
    imageinput = ImageInput("image/zone-test-1.tif", 0.3, 0.3)
    file_output = "output_class.csv"
    sys.stdout = open(file_output, 'w')
    sliding_window(imageinput)
    sys.stdout.close()

if __name__ == "__main__":
    main()
