---
aliases: python3, python2, python-3, python-2, python27, python-3-6, python-3-5, py
created_by: Guido van Rossum
display_name: Python
github_url: https://github.com/python
logo: python.png
related: language, ruby
released: February 20, 1991
short_description: Python is a dynamically typed programming language.
topic: python
url: https://www.python.org/
wikipedia_url: https://en.wikipedia.org/wiki/Python_(programming_language)
---
Python is a dynamically typed programming language designed by Guido van Rossum. Much like the programming language Ruby, Python was designed to be easily read by programmers. Because of its large following and many libraries, Python can be implemented and used to do anything from webpages to scientific research.
import cv2
import numpy as np
import glob
import sys

FNAME = 'digits.npz'

def machineLearning():
    img = cv2.imread('images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    x = np.array(cells)
    train = x[:,:].reshape(-1,400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k,500)[:,np.newaxis]

    np.savez(FNAME,train=train,train_labels = train_labels)

def resize20(pimg):
    img = cv2.imread(pimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayResize = cv2.resize(gray,(20,20))
    ret, thresh = cv2.threshold(grayResize, 125, 255,cv2.THRESH_BINARY_INV)

    cv2.imshow('num',thresh)
    return thresh.reshape(-1,400).astype(np.float32)

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def checkDigit(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("option : train or test")
        exit(1)
    elif sys.argv[1] == 'train':
        machineLearning()
    elif sys.argv[1] == 'test':
        train, train_labels = loadTrainData(FNAME)

        saveNpz = False
        for fname in glob.glob('images/num*.png'):
            test = resize20(fname)
            result = checkDigit(test, train, train_labels)

            print( result)

            k = cv2.waitKey(0)

            if k > 47 and k<58:
                saveNpz = True
                train = np.append(train, test, axis=0)
                newLabel = np.array(int(chr(k))).reshape(-1,1)
                train_labels = np.append(train_labels, newLabel,axis=0)


        cv2.destroyAllWindows()
        if saveNpz:
            print("a")
            np.savez(FNAME,train=train, train_labels=train_labels)
    else:
        print("unknow option")
print("a")
