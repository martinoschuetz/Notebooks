model
image
augment
with mask
    prediction
    for each masked image
build
heatmaps

conn.retrieve('image.augmentImages', message_level='error', imageTable=validTbl,
              casout={'name': 'valid_block', 'replace': True, 'blocksize': 128},
              cropList=[dict(sweepImage=True, x=0, y=0,
                             width=mask_width, height=mask_height, stepsize=step_size,
                             outputwidth=output_width,
                             outputheight=output_height,
                             mask=True)])
