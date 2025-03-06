"""
Usage:
 # From tensorflow/models/
 # Create train data:
 python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=train.record

 # Create test data:
 python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# tf.app.flags kullanımı kaldırıldı, argparse ile değiştirildi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_input', help='Path to the CSV input', required=True)
parser.add_argument('--output_path', help='Path to output TFRecord', required=True)
parser.add_argument('--image_dir', help='Path to images', required=True)
args = parser.parse_args()


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'airplane':
        return 1
    elif row_label == 'baseball bat':
        return 2
    elif row_label == 'bed-matress':
        return 3
    elif row_label == 'boat':
        return 4
    elif row_label == 'bus':
        return 5
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    file_path = os.path.join(path, group.filename)
    
    # Dosyanın var olup olmadığını kontrol et
    if not tf.io.gfile.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        # Dosyayı binary olarak aç ve oku
        with tf.io.gfile.GFile(file_path, 'rb') as fid:
            encoded_image = fid.read()
        
        # Resmi aç ve formatını kontrol et
        encoded_image_io = io.BytesIO(encoded_image)
        image = Image.open(encoded_image_io)
        
        # Transparency kanalını kaldırarak RGB'ye dönüştür (PNG için)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        
        width, height = image.size

        # Format belirleme (PIL ve dosya uzantısı kombinasyonu)
        if image.format:
            image_format = image.format.lower().encode('utf8')
        else:
            format_mapping = {
                '.jpg': b'jpeg',
                '.jpeg': b'jpeg',
                '.png': b'png'
            }
            file_extension = os.path.splitext(group.filename)[1].lower()
            image_format = format_mapping.get(file_extension, b'jpeg')

        filename = group.filename.encode('utf8')
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        
        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(f"Error details: {str(e)}")
        return None

def main():
    writer = tf.io.TFRecordWriter(args.output_path)
    path = os.path.join(os.getcwd(), args.image_dir)  # Mutlak yol oluştur
    examples = pd.read_csv(args.csv_input)
    grouped = split(examples, 'filename')
    
    error_files = []  # Hata veren dosyaları kaydetmek için

    for group in grouped:
        tf_example = create_tf_example(group, path)
        
        # Eğer create_tf_example None döndürürse, bu dosyayı atla
        if tf_example is None:
            print(f"Skipping file due to error: {group.filename}")
            error_files.append(group.filename)
            continue
        
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), args.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    
    # Hata veren dosyaları logla
    if error_files:
        print("\nThe following files were skipped due to errors:")
        for file in error_files:
            print(file)


if __name__ == '__main__':
    main()