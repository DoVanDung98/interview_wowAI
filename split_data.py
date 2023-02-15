import splitfolders

splitfolders.ratio("/home/vandung98/Desktop/wowAI/data_1",
                   output="data_processed_v3", seed=1337, ratio=(.7, .15, .15),
                   group_prefix=None)