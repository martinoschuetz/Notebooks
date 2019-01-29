def VGG(s):
    s.buildmodel(model=dict(name='VGG',replace=True),type='CNN')

    s.addlayer(model='VGG', name='data', replace=True,
              layer=dict(type='input',nchannels=4, width=28, height=28))
    #__________________________________________________________________________________________________
    #Block 1 
    s.addLayer(model='VGG', name='conv1', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=64, width=3, height=3, stride=1, init='MSRA', dropout=0.3), srcLayers=['data'])

    s.addLayer(model='VGG', name='bn1', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv1'])

    s.addLayer(model='VGG', name='conv2', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=64, width=3, height=3, stride=1, init='MSRA'), srcLayers=['bn1'])

    s.addLayer(model='VGG', name='bn2', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv2'])

    s.addLayer(model='VGG', name='pool1', replace=True,
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'), 
               srcLayers=['bn2'])
    # __________________________________________________________________________________________________
    # Block2
    s.addLayer(model='VGG', name='conv3', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=128, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['pool1'])

    s.addLayer(model='VGG', name='bn3', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv3'])

    s.addLayer(model='VGG', name='conv4', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=128, width=3, height=3, stride=1, init='MSRA'), srcLayers=['bn3'])

    s.addLayer(model='VGG', name='bn4', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv4'])

    s.addLayer(model='VGG', name='pool2', replace=True,
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'), 
               srcLayers=['bn4'])
    # __________________________________________________________________________________________________
    # Block3
    s.addLayer(model='VGG', name='conv5', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=256, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['pool2'])

    s.addLayer(model='VGG', name='bn5', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv5'])

    s.addLayer(model='VGG', name='conv6', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=256, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['bn5'])

    s.addLayer(model='VGG', name='bn6', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv6'])

    s.addLayer(model='VGG', name='conv7', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=256, width=3, height=3, stride=1, init='MSRA'), srcLayers=['bn6'])

    s.addLayer(model='VGG', name='bn7', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv7'])

    s.addLayer(model='VGG', name='pool3', replace=True,
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'), 
               srcLayers=['bn7'])
    # __________________________________________________________________________________________________
    # Block4
    s.addLayer(model='VGG', name='conv8', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['pool3'])

    s.addLayer(model='VGG', name='bn8', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv8'])

    s.addLayer(model='VGG', name='conv9', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['bn8'])

    s.addLayer(model='VGG', name='bn9', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv9'])

    s.addLayer(model='VGG', name='conv10', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA'), srcLayers=['bn9'])

    s.addLayer(model='VGG', name='bn10', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv10'])

    s.addLayer(model='VGG', name='pool4', replace=True,
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'), 
               srcLayers=['bn10'])
    # __________________________________________________________________________________________________
    # Block5
    s.addLayer(model='VGG', name='conv11', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['pool4'])

    s.addLayer(model='VGG', name='bn11', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv11'])

    s.addLayer(model='VGG', name='conv12', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA', dropout=0.4), srcLayers=['bn11'])

    s.addLayer(model='VGG', name='bn12', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv12'])

    s.addLayer(model='VGG', name='conv13', replace=True, 
               layer=dict(type='convolution', act='IDENTITY', includeBias=False, 
                          nFilters=512, width=3, height=3, stride=1, init='MSRA'), srcLayers=['bn12'])

    s.addLayer(model='VGG', name='bn13', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['conv13'])

    s.addLayer(model='VGG', name='pool5', replace=True,
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max', dropout=0.5), 
               srcLayers=['bn13'])
    # #__________________________________________________________________________________________________
    # Dense layer 
    s.addLayer(model='VGG', name='fc1',  replace=True,
               layer=dict(type='fullconnect',n=512, act='IDENTITY', includeBias = False, init='MSRA', dropout = 0.5), 
               srcLayers=['pool5'])

    s.addLayer(model='VGG', name='bn14', replace=True, layer=dict(type='BATCHNORM', act='relu'), srcLayers=['fc1'])
    # #__________________________________________________________________________________________________
    # softmax output layer
    s.addLayer(model='VGG', name='outlayer', replace=True,
               layer=dict(type='output',n=10,act='softmax', init='MSRA'), 
               srcLayers=['bn14'])