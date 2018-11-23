#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from swat import *


class Image:
    def __init__(self, sess, path=None, name='_ImageData_', blocksize=64):
        if not sess.queryactionset('image')['image']:
            sess.loadactionset('image')
        self.path = path
        self.sess = sess

        self.dataTbl = None
        self.trainTbl = None
        self.validTbl = None

        self.dataTbl_Crop = None
        self.trainTbl_Crop = None
        self.validTbl_Crop = None

        self.dataTbl_Resize = None
        self.trainTbl_Resize = None
        self.validTbl_Resize = None

        self.dataTbl_Patches = None
        self.trainTbl_Patches = None
        self.validTbl_Patches = None
        if path is not None:
            self.load(path, name=name, blocksize=blocksize)

    def load(self, path, name='_ImageData_', blocksize=64, **kwargs):
        sess = self.sess
        sess.image.loadimages(
            casout=dict(name=name, replace=True, blocksize=blocksize),
            distribution=dict(type='random'), recurse=True, labelLevels=-1,
            path=path, **kwargs)
        self.dataTbl = sess.CASTable(name)

    def train_test_split(self, test_rate=20):
        sess = self.sess
        if not sess.queryactionset('sampling')['sampling']:
            sess.loadactionset('sampling')

        sess.stratified(output=dict(casOut=dict(name="_ImageData_", replace=True),
                                    copyVars=['_image_', '_label_', '_path_']),
                        samppct=test_rate, samppct2=100 - test_rate,
                        table=dict(name='_ImageData_', groupby='_label_'))
        self.validTbl = sess.CASTable('_ImageData_', where='_partind_=1')
        self.trainTbl = sess.CASTable('_ImageData_', where='_partind_=2')

    def display(self, table='dataTbl', nimages=5):
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        if table not in ['dataTbl', 'trainTbl', 'validTbl']:
            raise ValueError("table must be one of the following: 'dataTbl','trainTbl','validTbl'")
        sess = self.sess

        table = eval('self.{}'.format(table))
        a = sess.table.fetch(sastypes=False,
                             table=table,
                             to=nimages)
        if nimages > 8:
            nrow = nimages // 8 + 1
            ncol = 8
        else:
            nrow = 1
            ncol = nimages
        fig = plt.figure(figsize=(16, 2 * nrow))

        for i in range(nimages):
            image = a.Fetch._image_[i]
            label = a.Fetch._label_[i]
            img = cv2.imdecode(np.fromstring(image, np.uint8), 1)
            # img = cv2.resize(img, (256, 256))
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.set_title('{}'.format(label))
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])
        plt.show()

    def crop(self, input_tbl='dataTbl', x=0, y=0, width=256, height=256,
             replace=False, output_tbl='Auto'):
        sess = self.sess

        if output_tbl is 'Auto':
            output_tbl = '{}_Crop'.format(input_tbl)
        if input_tbl not in ['dataTbl', 'trainTbl', 'validTbl']:
            raise ValueError("input_tbl must be one of the following: 'dataTbl','trainTbl','validTbl'")
        input_tbl = eval('self.{}'.format(input_tbl))

        if replace:
            sess.processimages(
                imageTable=input_tbl,
                casout={'name': input_tbl, 'replace': True},
                imagefunctions=dict(functionoptions=
                                    dict(functionType="GET_PATCH", x=x, y=y,
                                         width=width, height=height)))
        else:
            sess.processimages(
                imageTable=input_tbl,
                casout={'name': output_tbl, 'replace': True},
                imagefunctions=dict(functionoptions=
                                    dict(functionType="GET_PATCH", x=x, y=y,
                                         width=width, height=height)))
            if input_tbl is 'dataTbl':
                self.dataTbl_Crop = sess.CASTable(output_tbl)
            if input_tbl is 'trainTbl':
                self.trainTbl_Crop = sess.CASTable(output_tbl)
            if input_tbl is 'validTbl':
                self.validTbl_Crop = sess.CASTable(output_tbl)

    def resize(self, input_tbl='dataTbl', width=256, height=256,
               replace=False, output_tbl='Auto'):
        sess = self.sess

        if output_tbl is 'Auto':
            output_tbl = '{}_Resize'.format(input_tbl)
        if input_tbl not in ['dataTbl', 'trainTbl', 'validTbl']:
            raise ValueError("input_tbl must be one of the following: 'dataTbl','trainTbl','validTbl'")
        input_tbl = eval('self.{}'.format(input_tbl))

        if replace:
            sess.processimages(
                imageTable=input_tbl,
                casout={'name': input_tbl, 'replace': True},
                imagefunctions=dict(functionoptions=
                                    dict(functionType="RESIZE",
                                         width=width, height=height)))
        else:
            sess.processimages(
                imageTable=input_tbl,
                casout={'name': output_tbl, 'replace': True},
                imagefunctions=dict(functionoptions=
                                    dict(functionType="RESIZE",
                                         width=width, height=height)))
            if input_tbl is 'dataTbl':
                self.dataTbl_Resize = sess.CASTable(output_tbl)
            if input_tbl is 'trainTbl':
                self.trainTbl_Resize = sess.CASTable(output_tbl)
            if input_tbl is 'validTbl':
                self.validTbl_Resize = sess.CASTable(output_tbl)

    def patches(self, input_tbl='dataTbl', x=0, y=0, width=256, height=256,
                stepSize=None, outputWidth=None, outputHeight=None,
                replace=False, output_tbl='Auto'):
        sess = self.sess

        if output_tbl is 'Auto':
            output_tbl = '{}_Patches'.format(input_tbl)
        if input_tbl not in ['dataTbl', 'trainTbl', 'validTbl']:
            raise ValueError("input_tbl must be one of the following: 'dataTbl','trainTbl','validTbl'")
        input_tbl = eval('self.{}'.format(input_tbl))

        if stepSize is None:
            stepSize = width
        if outputWidth is None:
            outputWidth = width
        if outputHeight is None:
            outputHeight = height

        croplist = [dict(sweepImage=True, x=x, y=y,
                         width=width, height=height,
                         stepSize=stepSize,
                         outputWidth=outputWidth,
                         outputHeight=outputHeight)]
        if replace:
            sess.augmentImages(
                imageTable=input_tbl,
                casout={'name': input_tbl, 'replace': True},
                cropList=croplist)
        else:
            sess.augmentImages(
                imageTable=input_tbl,
                casout={'name': output_tbl, 'replace': True},
                cropList=croplist)

            if input_tbl is 'dataTbl':
                self.dataTbl_Patches = sess.CASTable(output_tbl)
            if input_tbl is 'trainTbl':
                self.trainTbl_Patches = sess.CASTable(output_tbl)
            if input_tbl is 'validTbl':
                self.validTbl_Patches = sess.CASTable(output_tbl)

    def summary(self, table='dataTbl'):
        sess = self.sess

        table = eval('self.{}'.format(table))
        print(sess.image.summarizeimages(imageTable=table))

    def freq(self, table='dataTbl'):
        table = eval('self.{}'.format(table))
        table.freq(inputs='_input_')
