# Oluşturduğumuz CSV dosyası üzeriden uygun klasörleme yapısı oluşturma
import pandas as pd
import shutil
import os
from glob import glob

# Veri seti
CSV_FILE = 'C:/Users/User/Desktop/CNN/dataset.csv'
df = pd.read_csv(CSV_FILE)

# Fazlalıklardan kurtul
TRAIN_SIZE = 8500
TEST_SIZE = 1000

# TEST + TRAIN kadarını al
df = df.groupby('Finding Labels').head(TRAIN_SIZE + TEST_SIZE)

# TRAIN kadarını baştan ayır
train_df = df.groupby('Finding Labels').head(TRAIN_SIZE)
# TEST kadarını sondan ayır
test_df = df.groupby('Finding Labels').tail(TEST_SIZE)

IMG_PATH = 'C:/Users/User/Desktop/CNN/archive/images*/images/*.png'
TRAIN_IMG_DESTINATION = 'C:/Users/User/Desktop/CNN/archive/dataset4/training_set'
TEST_IMG_DESTINATION = 'C:/Users/User/Desktop/CNN/archive/dataset4/test_set'

def copy_images_from_dataframe(df, img_path, destination):
    img_glob = glob(img_path)
    full_img_paths = {os.path.basename(x):x for x in img_glob}
    df['full_path'] = df['Image Index'].map(full_img_paths.get)
    df['full_path'] = df['full_path'].str.replace("\\", "/", regex=True)
    
    df = df.reset_index()
    for _, row in df.iterrows():
        disease_path = os.path.join(destination, row['Finding Labels']).replace("\\", "/")
        shutil.copy(row['full_path'], disease_path)

# Train verilerini taşı
copy_images_from_dataframe(train_df, IMG_PATH, TRAIN_IMG_DESTINATION)

# Test verilerini taşı
copy_images_from_dataframe(test_df, IMG_PATH, TEST_IMG_DESTINATION)















