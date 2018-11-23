import os

from swat import *
from dl_api import VGG16, Model
from dl_api.images import Image

# for item in sys.path:print(item)
sess = CAS('snap001.unx.sas.com', 14931, nworkers=3)

sess.tableinfo()

my_images = Image(sess)

my_images.load(path='/dept/cas/leliuz/WildTrack/Data0612/binary_fp')
tr_img, te_img = my_images.train_test_split(test_rate=80)
my_images.summary()
my_images.freq()
my_images.display(nimages=20)
my_images.resize(width=1024, height=1024)
my_images.summary()
my_images.freq()
my_images.display(nimages=20)
my_images.crop(x=32, y=32, width=512, height=512)
my_images.summary()
my_images.freq()
my_images.display(nimages=20)
my_images.patches(x=0, y=0, width=128, height=128)
my_images.summary()
my_images.freq()
my_images.display(nimages=20)
tr_img.display()

sess.endsession()
