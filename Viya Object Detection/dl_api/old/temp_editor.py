nloOpts = dict(
    mode=dict(type='SYNCHRONOUS'),

    algorithm=dict(method='momentum',
                   clipgradmin=-1000,
                   clipgradmax=1000,
                   learningRate=0.01,
                   lrpolicy='step',
                   stepsize=15,
                   useLocking=True
                   ),
    dropoutType='inverted',
    miniBatchSize=2,
    maxEpochs=5,
    regL2=0.0005,
    logLevel=3
    # ,snapshotfreq=10
)
model.fit(data=tr_img,
          targetOrder='ascending',
          nThreadsReduce=8,
          bufferSizeReduce=1.0,
          optimizer=nloOpts
          )
