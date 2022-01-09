import splitfolders as sf

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
sf.ratio('./gls',
         output='./gls_train_val',
         seed=1337,
         ratio=(.8, .2),
         group_prefix=None)

'''
# 剩下的都留给train
# Split `val/test`` with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
sf.fixed("input_folder",
         output="output",
         seed=1337,
         fixed=(100, 100),
         oversample=False,
         group_prefix=None)
'''
