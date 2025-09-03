mor_i = [0.138000,
0.474451, # 0.3 looks pretty
0.473873,
0.556474,
0.670174,
0.751476,
0.822139,
0.868768,
0.893231,
0.886591,
0.854912,
0.810000]#lie
mor_i.reverse()




mor_i = [-0.822732,
0.021566,
0.516586,
0.466559,
0.541117,
0.660657,
0.745429,
0.811658,
0.859461,
0.885600]




mor_i = [
    0.466559,
0.521040,
0.447128,
0.487799,
0.616081,
0.541359,
0.456700,
0.598822,
0.591357,
0.413925,
0.415947,
0.412850,
0.344679,
0.343321,
0.414953,
0.346097,
0.215924,
0.159141,
0.138071,
0.044812
]
#mor_i.reverse()
#mor_i = mor_i[:-1]
import matplotlib.pyplot as plt 
x = [64 + i * 10 for i in range (0,20)]
plt.plot(x, mor_i)
#plt.plot(len(mor_i)-3, mor_i[-3], 'ro', label = 'no significant autocorrelation')  
#plt.plot(len(mor_i)-1, mor_i[-1], 'ro')
plt.xlabel('n')
plt.ylabel('morans_i')
plt.legend()
plt.title('autocorrelation of blocks that are size 2^n')
plt.show()
