import tensorflow as tf  
#from scipy import mat
import csv
Max_TemperatureC,Mean_TemperatureC,Min_TemperatureC,Dew_PointC,MeanDew_PointC,Min_DewpointC,Max_Humidity,Mean_Humidity,Min_Humidity,Max_Sea_Level_PressurehPa,Mean_Sea_Level_PressurehPa,Min_Sea_Level_PressurehPa,Max_VisibilityKm,Mean_VisibilityKm,Min_VisibilitykM,Max_Wind_SpeedKmh,Mean_Wind_SpeedKmh,Max_Gust_SpeedKmh,Precipitationmm,CloudCover,Events,WindDirDegrees = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
i=0

def is_floatable(s):
    try:
        temp = float(s)
        return True
    except:
        return False

with open('data.csv', newline='') as csvfile:   # verisetinin sütun sütun okunduğu kısım
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(row['CET'],"-->",i)
        Max_TemperatureC.append(float(row['Max TemperatureC']))
        Mean_TemperatureC.append(float(row['Mean TemperatureC']))
        Min_TemperatureC.append(float(row['Min TemperatureC']) if row['Min TemperatureC'] is not None else 0)
        Dew_PointC.append(float(row['Dew PointC']))
        MeanDew_PointC.append(float(row['MeanDew PointC']))
        Min_DewpointC.append(float(row['Min DewpointC']))
        Max_Humidity.append(float(row['Max Humidity']))
        Mean_Humidity.append(float(row[' Mean Humidity']))
        Min_Humidity.append(float(row[' Min Humidity']))
        Max_Sea_Level_PressurehPa.append(float(row[' Max Sea Level PressurehPa']))
        Mean_Sea_Level_PressurehPa.append(float(row[' Mean Sea Level PressurehPa']))
        Min_Sea_Level_PressurehPa.append(float(row[' Min Sea Level PressurehPa']))
        Max_VisibilityKm.append(float(row[' Max VisibilityKm']) if is_floatable(row[' Max VisibilityKm']) is True else 0)
        Mean_VisibilityKm.append(float(row[' Mean VisibilityKm']) if is_floatable(row[' Mean VisibilityKm']) is True else 0)
        Min_VisibilitykM.append(float(row[' Min VisibilitykM']) if is_floatable(row[' Min VisibilitykM']) is True else 0)
        Max_Wind_SpeedKmh.append(float(row[' Max Wind SpeedKm/h']) if is_floatable(row[' Max Wind SpeedKm/h']) is True else 0)
        Mean_Wind_SpeedKmh.append(float(row[' Mean Wind SpeedKm/h']) if is_floatable(row[' Mean Wind SpeedKm/h']) is True else 0)
        Max_Gust_SpeedKmh.append(float(row[' Max Gust SpeedKm/h']) if is_floatable(row[' Max Gust SpeedKm/h']) is True else 0)
        Precipitationmm.append(float(row['Precipitationmm']))
        CloudCover.append(float(row[' CloudCover']) if is_floatable(row[' CloudCover']) is True else 0)
        Events.append(str(row[' Events']) if is_floatable(row[' Events']) is True else 0)
        WindDirDegrees.append(float(row['WindDirDegrees']))
        

inputs= [[0] * 2] * (len(Max_TemperatureC)-2)   # çift boyutlu matris oluşturuldu.
outputs=[[]] * (len(Max_TemperatureC)-2)   # çift boyutlu matris oluşturuldu.
max = 0     # Max_TemperatureC sütununu normalize edebilmek için max, min değerleri bulunacak
min = Max_TemperatureC[0]
for i in range(len(Max_TemperatureC) - 2):  # Max_TemperatureC'deki ilk 2 değer hariç gezerek max, min değerleri bulunur.
    if(Max_TemperatureC[i]>max):
        max=Max_TemperatureC[i]
    if(Max_TemperatureC[i]<min):
        min=Max_TemperatureC[i]
    inputs[i][0]= Max_TemperatureC[i]
    inputs[i][1]= Max_TemperatureC[i+1]

for i in range(len(Max_TemperatureC) - 2):  # inputların normalizasyonun yapıldığı kısım
    inputs[i][0] = (inputs[i][0] - min) / (max-min)
    inputs[i][1] = (inputs[i][1] - min) / (max-min)

for i in range(2,len(Max_TemperatureC)):    # output normalizasyonun yapıldığı kısım
    outputs[i-2] =[(Max_TemperatureC[i]-min) /(max-min)] 


# yer tutucular, kuracağımız ağın veri noktasıdır. Bizim giriş ve çıkış verilerimiz için bir kapı noktası.
# iki parametre oluşturduk. Tensorflow'da veri ağ üzerinden matris şeklinde akar.
x_ = tf.placeholder(tf.float32, shape=[None,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None,1], name="y-input")

# gizli düğüm sayısı
hidden_nodes = 5

b_hidden = tf.Variable(tf.random_normal([hidden_nodes]), name="hidden_bias")
W_hidden = tf.Variable(tf.random_normal([2, hidden_nodes]), name="hidden_weights")
hidden = tf.sigmoid(tf.matmul(x_, W_hidden) + b_hidden)

W_output = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="output_weights")  # çıkış katmanının ağırlık matrisi
output = tf.sigmoid(tf.matmul(hidden, W_output))  # çıkış katmanının aktivasyon fonksiyonunu hesaplama

cross_entropy = tf.square(y_ - output)  # 0 ile 1 arasında hata oranlarını hesaplama. Basit, ama aynı zamanda çalışır.

loss = tf.reduce_mean(cross_entropy)  # kayıp değerleri
optimizer = tf.train.GradientDescentOptimizer(0.01)  # 0.1'lik bir "adımboyu" ile optimize etmek için bir degrade iniş alma(gradient descent)
train = optimizer.minimize(loss)  # optimize ediciyi eğitme kısmı

#Şimdiye kadar yaptığımız şey, tensorflow'u ağın yapısının ne olduğunu ve verilerin nasıl akacağını tanımladık.
#ancak ağ hala çalışmıyor ve henüz bir işlem yapılmadı. Bu yüzden ağı çalıştırmak için tensorflow oturumu başlatmamız gerekiyor.
#Gerçekleşen herhangi bir hesaplama bir oturum içinde gerçekleşecek, gerçekte yaptığımız tensorflow işlemi gerçekte oturumu başlatıp çalıştırdıktan sonra gerçekleşecek.
#Böylece oturuma başlamadan önce ilk önce veri setini oluşturmamıza izin verilir. Başka bir ikili kapı XOR kapısı olarak çalışmasını sağlayacağız.
#Tabii ki çok karmaşık şeyler yapabiliriz ama basitlik uğruna XOR geçidi gibi çalışmasını sağlayacağız.
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]
loss_list=[]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
        sess.run(train, feed_dict={x_: inputs, y_: outputs})
        loss_list.append(sess.run(loss, feed_dict={x_: inputs, y_: outputs}))
        if i % 100 == 0:
            print('Epoch ', i)
while True:
    inp = []
    inp.append(float(input("1.Maksimum sicakligi giriniz :")))
    inp.append(float(input("2.Maksimum sicakligi giriniz :")))

    inp[0] = (inp[0] - min) / (max-min)
    inp[1] = (inp[1] - min) / (max-min)
    predict = sess.run(output, feed_dict={x_: [inp]})
    predict = ((max-min) * predict) + min
    print("Tahmini maksimum sicaklik :",predict)
    Max_TemperatureC.pop(0)
    Max_TemperatureC.pop(0)     # sonuç kısmı ilk 2 değerden sonra başladığı için dizinin ilk 2 elemanı silindi.
    
    outputsNorm = outputs   # outputs ile aynı boyutta dizi oluşturuldu.
    for i in range(len(outputs)):
        outputsNorm[i] = float(outputs[i][0])*float(max-min) + float(min)
        # normalize edilen input değerlerinden gelen sonuç 0-1 arasındaydı.
        # bu tekrar eski normalize olmamış haline döndürmek için verisetinden bulunan min - max değerleri kullanılarak veri olması gereken aralığa döndürüldü.
        # normalizasyon formülü : (xi - xmin) / (xmax - xmin)
        # normalizasyonu tersine çevirmek için formül : xi * (max - min) + min

    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Hata Oranı')
    plt.show()
    x=Max_TemperatureC  # verisetinden alınan dizi farklı değişkene atandı.

    # 14.0000000002 cikan degeri int degerinden cikar, 10^15 ile carpip x[i] ekledim.
    y=[(outputsNorm[i] - int(outputsNorm[i]))*pow(10,14) + x[i] for i in range(0,len(x))]   # tahmin edilen değerler dizisi farklı değişkene atandı.
    
    for i in range(0,6808):     # gerçek değer ile tahmin edilen değer eşit değilse print verir.
        if x[i] != y[i]:
            print(i, "esit degil")
            print("x: {} y: {}".format(x[i], y[i]))

    plt.plot(range(0, 6808), x, label='Gerçek Değer', marker='*')
    plt.plot(range(0, 6808), y, label='Tahmin Edilen YSA Çıkışı')

    #plt.plot(range(0, 10), [4,5,7,3,7,2,4,8,6,0], label='Gerçek değerler', linewidth=1, marker='*')
    #plt.plot(range(0, 10), [7,45,4,2,6,8,9,0,4,5], label='Tahmin değerleri', marker='o', color='green', linewidth=1, linestyle='dashed')
    plt.legend()
    plt.xlabel('Örnek Sayısı')
    plt.ylabel('Sıcaklık Değeri')
    plt.savefig('Benim Grafiğim.png')
    plt.show()