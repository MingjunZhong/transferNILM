from Logger import log
import numpy as np
import pandas as pd


class ChunkDoubleSourceSlider(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5*10**5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold

    def check_lenght(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        t_size = 0

        for chunk in check_cvs:
            size = chunk.shape[0]
            t_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(t_size/10 ** 6))
        if t_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

        return t_size

    def feed_chunk(self):

        try:
            total_size
        except NameError:
            #global total_size
            total_size = ChunkDoubleSourceSlider.check_lenght(self)

        if total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            # iterations over csv file
            for chunk in data_frame:

                np_array = np.array(chunk)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                """
                if len(inputs) < self.batchsize:
                    while len(inputs) == self.batchsize:
                        inputs = np.append(inputs, 0)
                       targets = np.append(targets, 0)
                """

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):

                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                    tar = targets[excerpt + self.offset].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]


            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                    self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                    np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                tar = targets[excerpt + self.offset].reshape(-1, 1)

                yield inp, tar


class ChunkDoubleSourceSlider2(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkDoubleSourceSlider2.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                    tar = targets[excerpt + self.offset].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                tar = targets[excerpt + self.offset].reshape(-1, 1)

                yield inp, tar


class DoubleSourceProvider2(object):

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets):

        assert len(inputs) == len(targets)

        inputs = inputs.flatten()
        targets = targets.flatten()

        max_batchsize = inputs.size - 2 * self.offset

        if self.batchsize == -1:
            self.batchsize = len(inputs)

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]

            yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]),\
                  targets[excerpt + self.offset].reshape(-1, 1)


class DoubleSourceProvider3(object):

    def __init__(self, nofWindows, offset):

        self.nofWindows = nofWindows
        self.offset = offset

    def feed(self, inputs):

        inputs = inputs.flatten()
        max_nofw = inputs.size - 2 * self.offset

        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])

            yield inp

