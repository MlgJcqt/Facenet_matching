# adapted from davidsanberg/facenet
# author : Maelig Jacquet
# last version : 09.04.2021
# objective : Performing face alignment and calculating L2 distance between the embeddings of images from 2 folders


###
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
from PIL import Image
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()

parser.add_argument( 'image_files', type=str, nargs='+', help='Images to compare' )
parser.add_argument( '--image_size', type=int,
                     help='Image size (height, width) in pixels.', default=160 )
parser.add_argument( '--margin', type=int,
                     help='Margin for the crop around the bounding box (height, width) in pixels.', default=44 )
parser.add_argument( '--gpu_memory_fraction', type=float,
                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.01 )
parser.add_argument( '--model', type=str,
                     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                     default='../data/model/20180402-114759.pb' )
parser.add_argument( 'out_file', type=str,
                     help='Output scores list.', default='../output/results.csv' )

args = parser.parse_args()
outfile = args.out_file
out_error = os.path.dirname( outfile ) + "/"

dirimg1 = args.image_files[0]
dirimg2 = args.image_files[1]

if os.path.isdir( dirimg1 ) :
    listimg1 = []
    for f in os.listdir( dirimg1 ) :
        listimg1.append( dirimg1 + "/" + f )
else :
    listimg1 = [dirimg1]

if os.path.isdir( dirimg2 ) :
    listimg2 = []
    for f in os.listdir( dirimg2 ) :
        listimg2.append( dirimg2 + "/" + f )
else :
    listimg2 = [dirimg2]

listdef1 = []
listdef2 = []


def main(lst1, lst2) :
    print( "\nMATCHING ..." )

    print( "\n     [ ", len( images1 ), " ] processed images from folder 1" )
    print( "     [ ", len( images2 ), " ] processed images from folder 2" )
    print( "     [ ", (len( images1 )) * (len( images2 )), " ] comparisons to run\n" )

    with tf.Graph().as_default() :

        with tf.compat.v1.Session() as sess :

            # Load the model
            facenet.load_model( args.model )

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name( "input:0" )
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name( "embeddings:0" )
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name( "phase_train:0" )

            # Run forward pass to calculate embeddings
            ###### original compare.py modif

            emb1 = np.array( [] )
            emb2 = np.array( [] )

            feed_dict1 = {images_placeholder : images1, phase_train_placeholder : False}
            emb1 = sess.run( embeddings, feed_dict=feed_dict1 )

            feed_dict2 = {images_placeholder : images2, phase_train_placeholder : False}
            emb2 = sess.run( embeddings, feed_dict=feed_dict2 )

            # Create output folder if doesnt exist
            out_path = os.path.dirname( outfile )
            if not os.path.exists( out_path ) :
                os.makedirs( out_path )

            outf = open( outfile, "w+" )
            outf.write( "Image 1;Image 2;Score\n" )

            for (i, a) in zip( range( len( images1 ) ), lst1 ) :
                nom_img1 = os.path.basename( a )
                for (j, b) in zip( range( len( images2 ) ), lst2 ) :
                    nom_img2 = os.path.basename( b )
                    dist = np.sqrt( np.sum( np.square( np.subtract( emb1[i, :], emb2[j, :] ) ) ) )

                    outf.write( "%s;%s;%.3f\n" % (nom_img1, nom_img2, dist) )

            outf.close()

            print( '\n... END' )


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction) :
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps threshold
    factor = 0.709  # scale factor

    image1 = []
    image2 = []

    errdet = []
    err = []

    with tf.Graph().as_default() :
        gpu_options = tf.compat.v1.GPUOptions( per_process_gpu_memory_fraction=args.gpu_memory_fraction )
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto( gpu_options=gpu_options, log_device_placement=False ) )
        with sess.as_default() :
            pnet, rnet, onet = align.detect_face.create_mtcnn( sess, None )

    print( "\n" )

    with tqdm( total=len( image_paths ), desc='Loading and aligning images from %s' % (
            os.path.basename( os.path.dirname( image_paths[0] ) )) ) as pbar2 :
        img_list = []
        for image in image_paths :
            # img = misc.imread(os.path.expanduser(image), mode='RGB')
            try :
                img = np.array( Image.open( os.path.expanduser( image ) ) )
                img_size = np.asarray( img.shape )[0 :2]
                bounding_boxes, _ = align.detect_face.detect_face( img, minsize, pnet, rnet, onet, threshold, factor )
            except Exception :
                err.append( os.path.basename( image ) )

            else :
                if len( bounding_boxes ) < 1 :
                    errdet.append( os.path.basename( image ) )

                    continue

                try :
                    det = np.squeeze( bounding_boxes[0, 0 :4] )
                    bb = np.zeros( 4, dtype=np.int32 )
                    bb[0] = np.maximum( det[0] - margin / 2, 0 )
                    bb[1] = np.maximum( det[1] - margin / 2, 0 )
                    bb[2] = np.minimum( det[2] + margin / 2, img_size[1] )
                    bb[3] = np.minimum( det[3] + margin / 2, img_size[0] )
                    cropped = img[bb[1] :bb[3], bb[0] :bb[2], :]
                    cropped = Image.fromarray( np.uint8( cropped ) )
                    aligned = cropped.resize( (args.image_size, args.image_size), Image.ANTIALIAS )
                    # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    # prewhitened = facenet.prewhiten(aligned)
                    img_list.append( aligned )
                    pbar2.update( 1 )

                except IndexError :
                    err.append( os.path.basename( image ) )

                else :
                    if image_paths == listimg1 :
                        image1 = np.stack( img_list )
                        listdef1.append( image )

                    elif image_paths == listimg2 :
                        image2 = np.stack( img_list )
                        listdef2.append( image )

    # report errors
    if (len( err ) + len( errdet )) == 0 :
        print( "\n     [  0  ] errors, all images have been successfully processed" )
    else :
        print( "\n     [  ", len( err ), "  ] images could not be processed due to errors" )
        print( "     [  ", len( errdet ), "  ] images could not be processed because no face had been detected" )

    with open( out_error + "Errors.txt", "w+" ) as outerror :
        for a in err :
            outerror.write( "%s%s" % (a, "\n") )

    with open( out_error + "Not_Detected.txt", "w+" ) as outdetect :
        for b in errdet :
            outdetect.write( "%s%s" % (b, "\n") )

    return image1 if image_paths == listimg1 else image2


images1 = load_and_align_data( listimg1, args.image_size, args.margin, args.gpu_memory_fraction )
images2 = load_and_align_data( listimg2, args.image_size, args.margin, args.gpu_memory_fraction )

main( listdef1, listdef2 )

# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
# print (sys.argv)
