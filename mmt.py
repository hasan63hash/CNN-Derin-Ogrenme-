# Sadece ilgili hastalıklara ait bir csv oluştur
import pandas as pd

# Veri seti
CSV_FILE = 'archive/Data_Entry_2017.csv'
df = pd.read_csv(CSV_FILE)

# Sadece tek hastalık içeren verileri al
df.drop(df[df["Finding Labels"].str.contains('\|') == True].index, inplace = True)

# Sadece seçilen hastalıkları al
diseases = [
        "Infiltration",
        "No Finding"
    ]
df.drop(df[~df['Finding Labels'].isin(diseases)].index, inplace = True)

# Frekansları görüntüle
print(df["Finding Labels"].value_counts())

# Dataframe'i csv'ye kaydet
df.to_csv("dataset.csv", encoding='utf-8')


