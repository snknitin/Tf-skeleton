# Tf-skeleton


This repository constains a skeletal structure with multiple files that can make starting on different projects easy and quicker. This will be built following several tensorflow tutorials and my own experiences.


# How to pick 
https://www.oreilly.com/ideas/square-off-machine-learning-libraries



# Bag of Tricks

    num_examples       = .. # Dataset size
    num_parallel_calls = 16 # CPU threads to use for decoding
    prefetch_size      = 16 # Queue up multiple batches CPU side

    # ...

    # Infinitely loop the dataset, shuffling once per epoch (in memory). 
    # Safe to do when the dataset pipeline is currently light weight references 
    # to data such as integer id's of examples or string filenames...
    data = data.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=num_examples))

    # Turn our lightweight reference to data into an actual example. 
    # Image id -> Image / Filename -> Image.
    data = data.map(decode_example, num_parallel_calls=num_parallel_calls)

    # Stack decoded examples into constant sized batches.
    # Throwing away the remainder allows the pipeline to report a fixed sized batch size, 
    # aiding in model definition downstream.
    data = data.batch(batch_size, drop_remainder=True)

    # Queue up a number of batches on the CPU side
    data = data.prefetch(prefetch_size)

    # Queue up batches asynchronously onto the GPU
    # As long as there is a pool of batches CPU side a GPU prefetch of 1 is sufficient. 
    data = data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:%d' % (device_num)))
    
    
 We use these steps in this order within our greater pipeline as appropriate. In general any decode functions we apply are before batching and operate on single examples at a time.

Due to prefetching the first batch to end up going through the GPU will incur a lag as the light weight references are shuffled at the start of the epoch and the prefetch buffers is filled for the first time.

Decoding functions are in a weird middle ground where they are executed entirely CPU bound, but using tensorflow ops. Sometimes we want to integrate with non tensoflow functions during this stage, which can be accomplished by using a mapped function which calls a tf.py_func op within it. The following is a dual decoding example where we call a tf.py_func which opens a h5py archive and samples a random timeframe from an audio waveform and returns it as a numpy array.

    # Paths to unique speakers (in this example)
    data_paths = ['1.wav', '2.wav', ...]

    # Given the id of a speaker, load their audio file and extract a random timeframe
    def decode_example(speaker_idx):
        with h5py.File(data_paths[speaker_idx], mode='r') as archive:
            length = archive['waveform'].shape[0]
            idx    = np.random.randint(length-window_samples)
            return archive['waveform'][idx:idx+window_samples]

    # Call our python decode function, and perform type conversion on the result. 
    def sample_example(speaker_idx):
        waveform = tf.py_func(decode_example, (speaker_idx,), (tf.int16,))
        waveform = tf.cast(waveform, tf.float32) / 32767.
        waveform = tf.reshape(waveform, shape=(-1,))

        return waveform, speaker_idx

    # Our input pipeline now goes from Speaker ID -> (Waveform Sample, Speaker ID) pairs.
    data = data.map(sample_example, num_parallel_calls=num_parallel_calls)
