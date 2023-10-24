import os
import pickle
import tensorflow as tf

directory = 'mystuff/data/deepset_toy'
output_file = 'deepset_dataset.tfrecords'


def main():
    with open(os.path.join(directory, output_file), 'a+') as out_file:
        writer = tf.io.TFRecordWriter(out_file.name)
        for i in range(1, 50):
            for filename in os.listdir(directory):
                if filename.endswith(f'{i}.pkl'):
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'rb') as in_file:
                        data, loss, hat = pickle.load(in_file)
                        for j in range(len(data)):
                            data_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(data[j].astype('float32')))
                            loss_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(loss[j]))
                            hat_bytes = tf.io.serialize_tensor(tf.convert_to_tensor(hat[j]))

                            example = tf.train.Example(features=tf.train.Features(feature={
                                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_bytes.numpy()])),
                                'loss': tf.train.Feature(bytes_list=tf.train.BytesList(value=[loss_bytes.numpy()])),
                                'hat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hat_bytes.numpy()]))
                            }))
                            writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
