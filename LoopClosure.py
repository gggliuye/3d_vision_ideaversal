import numpy as np
import cv2
import pickle
from pathlib import Path

class LoopClosure:
    """ class of methods of LoopClosure by bag of words"""
    image_signatures = []

    def __init__(self):
        self.detection = cv2.xfeatures2d.SURF_create(300, extended = False)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.bow_extract = cv2.BOWImgDescriptorExtractor(self.detection, self.matcher)

    def set_vocabulary(self, file_dictionary):
        try:
            self.dictionary = np.load(file_dictionary)
            self.bow_extract.setVocabulary( dictionary )
        except ValueError:
            print("Oops!  That was no valid file address.  Try again...")

    def set_vocabulary_by_train(self, FileList):
        BOW = cv2.BOWKMeansTrainer(100)
        for ImageName in FileList:
            img = np.load(ImageName)
            kp, des = self.detection.detectAndCompute(img,None)
            BOW.add(des)
        try:
            self.dictionary = BOW.cluster()
            self.bow_extract.setVocabulary( self.dictionary )
            print( '--- train finished ---')
        except ValueError:
            print("Oops!  Something went wrong.  Try again...")

    def save_vocabulary(self, file_dictionary):
        try:
            np.save(file_dictionary, self.dictionary)
        except ValueError:
            print("Oops!  Something went wrong.  Try again...")

    def get_signature(self, image):
        kp = self.detection.detect(image)
        kp, des = self.detection.compute(image,kp)
        inp = self.bow_extract.compute(image,kp)
        return inp

    def load_image_signatures(self, filedict):
        if Path(filedict).exists():
            with open(filedict, 'rb') as f:
                self.image_signatures = pickle.load(f)
        else:
            print("Oops!  No such file.  Try again...")

    def save_image_signatures(self, filedict):
        if Path(filedict).exists():
            with open(filedict, 'wb') as f:
                pickle.dump(self.image_signatures, f)
        else:
            print("Oops!  No such file.  Try again...")

    def loop_closure_detection(self, inp_image, image_num,filedict, threshold=0.08):
        detected = False
        num = []
        for sign in self.image_signatures:
            dis = np.linalg.norm(sign["inp"]-inp_image)
            if dis < threshold:
                detected = True
                num.append(sign["num"])

        if not detected:
            sign = {}
            sign["inp"] = inp_image
            sign["num"] = image_num
            self.image_signatures.append(sign)
            return detected, num
        else:
            return detected, num
