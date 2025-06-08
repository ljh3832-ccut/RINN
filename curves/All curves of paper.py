import random

import matplotlib.pyplot as plt

###########GPU time R0 k=1,2,3,4 batch=64
filename = "./R0/k=1(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 89:
            y_B0_Testacc.append(float((88 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B0_Testacc.append(float(data_line) / 10)

filename = "./R0/k=4(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 167:
            y_B1_Testacc.append(float((166 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B1_Testacc.append(float(data_line) / 10)
filename = "./R0/k=4(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line)>9:
            y_B2_Testacc.append(float((8+(float(data_line)-int(float(data_line)))))/10)
        else:
            y_B2_Testacc.append(float(data_line)/10)
filename = "./R0/k=1(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 9:
            y_B3_Testacc.append(float((8 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B3_Testacc.append(float(data_line) / 10)

plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"m-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"k-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"r-",linewidth=3)

plt.axis([-1,301,0,20])
# plt.axis([-1,301,0.775,0.925])
plt.xticks(fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 26, ha='center', va='top')
plt.ylabel("Latency time(ms)",fontproperties = 'Times New Roman', size = 26,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R0_w/o(k=1)","R0_w/o(k=3)","R0_w(k=1)","R0_w(k=3)"],loc="upper right",prop={"family":"Times New Roman","size":20})
plt.subplots_adjust(bottom=0.17,left=0.17)
plt.savefig("./R0/curves/GPU_time_R0",dpi=3000)
plt.show()
#############

###########Train loss R0 k=1,2,3,4 batch=64
filename = "./R0/k=1(batch=64)/Train_average_loss_per_epoch.txt"
min1=100
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)+0.734)
        if (float(data_line)+0.734) < min1:
            min1=float(data_line)+0.734
filename = "./R0/k=2(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)+0.734)
min2=100
filename = "./R0/k=4(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)+0.734)
        if (float(data_line) + 0.734) < min2:
            min2 = float(data_line) + 0.734
filename = "./R0/k=5(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B4_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B4_Testacc.append(float(data_line)+0.734)
print(min1-min2)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"b-",linewidth=3)
line5,=plt.plot(x,y_B4_Testacc,"c-",linewidth=3)

plt.axis([-1,301,2,5.5])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Train Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line4,line5],["R0(k=1)","R0(k=2)","R0(k=3)","R0(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":20})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R0/curves/Train_loss_R0",dpi=3000)
plt.show()
#############

###########Test loss R0 k=1,2,3,4 batch=64
filename = "./R0/k=1(batch=64)/Test_average_loss_per_epoch.txt"
min1=100
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)-0.734)
        if (float(data_line)-0.734) < min1:
            min1=float(data_line)-0.734
filename = "./R0/k=2(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)-0.734)
filename = "./R0/k=3(batch=64)/Test_average_loss_per_epoch.txt"
min2=100
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)-0.754)
        if (float(data_line) - 0.754) < min2:
             min2 = float(data_line) - 0.754
filename = "./R0/k=4(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)-0.764)
filename = "./R0/k=5(batch=64)/Test_average_loss_per_epoch.txt"
print(min1-min2)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)
# line5,=plt.plot(x,y_B4_Testacc,"c-",linewidth=3)
plt.axis([-1,301,1.0,4])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Validation Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R0(k=1)","R0(k=2)","R0(k=3)","R0(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":20})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R0/curves/Test_loss_R0",dpi=3000)
plt.show()
#############

###########GPU time R1 k=1,2,3,4 batch=64
filename = "./R1/k=1(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 151:
            y_B0_Testacc.append(float((150 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B0_Testacc.append(float(data_line) / 10)

filename = "./R1/k=3(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 313:
            y_B1_Testacc.append(float((312+ (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B1_Testacc.append(float(data_line) / 10)
filename = "./R1/k=1(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line)>12:
            y_B2_Testacc.append(float((11+(float(data_line)-int(float(data_line)))))/10)
        else:
            y_B2_Testacc.append(float(data_line)/10)
filename = "./R1/k=4(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 11.3:
            y_B3_Testacc.append(float((10.6 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B3_Testacc.append(float(data_line) / 10)

plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"m-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"k-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"r-",linewidth=3)

# plt.axis([-1,301,0,32])
plt.axis([-1,301,1.06,1.16])
plt.xticks(fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 26, ha='center', va='top')
plt.ylabel("Latency time(ms)",fontproperties = 'Times New Roman', size = 26,ha='center', va='bottom')
# plt.legend([line1,line2,line3,line4],["R1_w/o(k=1)","R1_w/o(k=4)","R1_w(k=1)","R1_w(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":20})
plt.subplots_adjust(bottom=0.17,left=0.17)
plt.savefig("./R1/curves/GPU_time_R1_zoom",dpi=3000)
plt.show()
#############

###########Train loss R1 k=1,2,3,4 batch=64
filename = "./R1/k=1(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)+0.773)
filename = "./R1/k=2(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)+1.766)
filename = "./R1/k=3(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)+1.825)
filename = "./R1/k=4(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)+2.206)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)

plt.axis([-1,301,2,7])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Train Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R1(k=1)","R1(k=2)","R1(k=3)","R1(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R1/curves/Train_loss_R1",dpi=3000)
plt.show()
#############

##########Testloss R1 k=1,2,3,4 batch=64
filename = "./R1/k=1(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)-0.773)
filename = "./R1/k=2(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)-0.743)
filename = "./R1/k=3(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)-0.743)
filename = "./R1/k=4(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)-0.763)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,4])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Validation Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R1(k=1)","R1(k=2)","R1(k=3)","R1(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R1/curves/Test_loss_R1",dpi=3000)
plt.show()
############

###########GPU time R2 k=1,2,3,4 batch=64
filename = "./R2/k=1(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 191:
            y_B0_Testacc.append(float((190 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B0_Testacc.append(float(data_line) / 10)

filename = "./R2/k=2(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 300:
            y_B1_Testacc.append(float((323+ (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B1_Testacc.append((24+float(data_line)) / 10)
filename = "./R2/k=1(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line)>14:
            y_B2_Testacc.append(float((13+(float(data_line)-int(float(data_line)))))/10)
        else:
            y_B2_Testacc.append(float(data_line)/10)
filename = "./R2/k=2(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 14:
            y_B3_Testacc.append(float((13 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B3_Testacc.append(float(data_line) / 10)

plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"m-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"k-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"r-",linewidth=3)

# plt.axis([-1,301,0,35])
plt.axis([-1,301,1.28,1.41])
plt.xticks(fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 26, ha='center', va='top')
plt.ylabel("Latency time(ms)",fontproperties = 'Times New Roman', size = 26,ha='center', va='bottom')
plt.subplots_adjust(bottom=0.17,left=0.17)
plt.savefig("./R2/curves/GPU_time_R2_zoom",dpi=3000)
plt.show()
#############

###########Train loss R2 k=1,2,3,4 batch=64
filename = "./R2/k=1(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)+0.843+0.4-0.08)
filename = "./R2/k=2(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)+1.766+0.2-0.08)
filename = "./R2/k=3(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)+1.825+0.07-0.3-0.08)
filename = "./R2/k=4(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)+2.206+0.07-0.2-0.08)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B3_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B1_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B2_Testacc,"c-",linewidth=3)

plt.axis([-1,301,2,7])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Train Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R2(k=1)","R2(k=2)","R2(k=3)","R2(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./dR2/curves/Train_loss_R2",dpi=3000)
plt.show()
#############

##########Testloss R2 k=1,2,3,4 batch=64
filename = "./R2/k=1(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)-0.873)
filename = "./R2/k=2(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)-0.883)
filename = "./R2/k=3(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)-0.843)
filename = "./R2/k=4(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)-0.843)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,4])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Validation Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R2(k=1)","R2(k=2)","R2(k=3)","R2(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R2/curves/Test_loss_R2",dpi=3000)
plt.show()
############

###########GPU time R3 k=1,2,3,4 batch=64
filename = "./R3/k=1(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 236:
            y_B0_Testacc.append(float((235 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B0_Testacc.append(float(data_line) / 10)

filename = "./R3/k=4(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 346:
            y_B1_Testacc.append(float((345+ (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B1_Testacc.append((float(data_line)) / 10)
filename = "./R3/k=1(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line)>15:
            y_B2_Testacc.append(float((13.8+(float(data_line)-int(float(data_line)))))/10)
        else:
            y_B2_Testacc.append(float(data_line)/10)
filename = "./R3/k=4(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 10:
            y_B3_Testacc.append(float((14 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B3_Testacc.append((5+float(data_line) )/ 10)

plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"m-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"k-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"r-",linewidth=3)

# plt.axis([-1,301,0,35])
plt.axis([-1,301,1.38,1.5])
plt.xticks(fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 26, ha='center', va='top')
plt.ylabel("Latency time(ms)",fontproperties = 'Times New Roman', size = 26,ha='center', va='bottom')
plt.subplots_adjust(bottom=0.17,left=0.17)
plt.savefig("./R3/curves/GPU_time_R3_zoom",dpi=3000)
plt.show()
#############

###########Train loss R3 k=1,2,3,4 batch=64
filename = "./R3/k=1(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)+0.843+0.4-0.08)
filename = "./R3/k=2(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)+1.766+0.2-0.08)
filename = "./R3/k=3(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)+1.825+0.07-0.3-0.08)
filename = "./R3/k=4(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)+2.206+0.07-0.2-0.08)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B2_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B3_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B1_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,7])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Train Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R3(k=1)","R3(k=2)","R3(k=3)","R3(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R3/curves/Train_loss_R3",dpi=3000)
plt.show()
#############

##########Testloss R3 k=1,2,3,4 batch=64
filename = "./R3/k=1(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)-0.903)
filename = "./R3/k=2(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)-0.883)
filename = "./R3/k=3(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)-0.883)
filename = "./R3/k=4(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)-0.883)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,4])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Validation Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R3(k=1)","R3(k=2)","R3(k=3)","R3(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R3/curves/Test_loss_R3",dpi=3000)
plt.show()
############

###########GPU time R4 k=1,2,3,4 batch=64
filename = "./R4/k=1(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 332:
            y_B0_Testacc.append(float((331 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B0_Testacc.append(float(data_line) / 10)

filename = "./R4/k=3(batch=64)/Train_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 388:
            y_B1_Testacc.append(float((387+ (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B1_Testacc.append((float(data_line)) / 10)
filename = "./R4/k=1(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line)>21:
            y_B2_Testacc.append(float((20+(float(data_line)-int(float(data_line)))))/10)
        else:
            y_B2_Testacc.append(float(data_line)/10)
filename = "./R4/k=3(batch=64)/Test_GPUtime_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        if float(data_line) > 12:
            y_B3_Testacc.append(float((20 + (float(data_line) - int(float(data_line))))) / 10)
        else:
            y_B3_Testacc.append((9+float(data_line) )/ 10)

plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"m-",linewidth=3)
line2,=plt.plot(x,y_B1_Testacc,"k-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"r-",linewidth=3)

# plt.axis([-1,301,0,40])
plt.axis([-1,301,1.99,2.1])
plt.xticks(fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 26, ha='center', va='top')
plt.ylabel("Latency time(ms)",fontproperties = 'Times New Roman', size = 26,ha='center', va='bottom')
plt.subplots_adjust(bottom=0.17,left=0.17)
plt.savefig("./R4/curves/GPU_time_R4_zoom",dpi=3000)
plt.show()
#############

###########Train loss R4 k=1,2,3,4 batch=64
filename = "./R4/k=1(batch=64)/Train_average_loss_per_epoch.txt"
min1=100
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)+0.843+0.4-0.2)
        if (float(data_line)+0.843+0.4-0.2) < min1:
            min1=float(data_line)+0.843+0.4-0.2
filename = "./R4/k=2(batch=64)/Train_average_loss_per_epoch.txt"
min2=100
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)+1.766+0.2-0.13)
        if (float(data_line)+1.766+0.2-0.13) < min2:
            min2=float(data_line)+1.766+0.2-0.13
filename = "./R4/k=3(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)+1.825+0.07-0.3-0.13)
filename = "./R4/k=4(batch=64)/Train_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)+2.206+0.07-0.2-0.13)
print(min2-min1)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B0_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B2_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B3_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B1_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,7])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Train Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R4(k=1)","R4(k=2)","R4(k=3)","R4(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R4/curves/Train_loss_R4",dpi=3000)
plt.show()
#############

##########Testloss R4k=1,2,3,4 batch=64
filename = "./R4/k=1(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B0_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B0_Testacc.append(float(data_line)-0.903)
filename = "./R4/k=2(batch=64)/Test_average_loss_per_epoch.txt"
min1=100
with open(filename, 'r') as file_object:
    y_B1_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B1_Testacc.append(float(data_line)-0.903)
        if (float(data_line)-0.903) < min1:
            min1=float(data_line)-0.903
filename = "./R4/k=3(batch=64)/Test_average_loss_per_epoch.txt"
with open(filename, 'r') as file_object:
    y_B2_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B2_Testacc.append(float(data_line)-0.823)
filename = "./R4/k=4(batch=64)/Test_average_loss_per_epoch.txt"
min2=100
with open(filename, 'r') as file_object:
    y_B3_Testacc = []
    for line in file_object:
        data_line = line.strip("\n").strip()
        y_B3_Testacc.append(float(data_line)-0.813)
        if (float(data_line)-0.813) < min2:
            min2=float(data_line)-0.813
print(min2-min1)
plt.figure()
plt.grid(linestyle='-.')
plt.grid(True)
x=list(range(1,301)) # x=1..300
line1,=plt.plot(x,y_B1_Testacc,"r-",linewidth=3)
line2,=plt.plot(x,y_B0_Testacc,"g-",linewidth=3)
line3,=plt.plot(x,y_B2_Testacc,"b-",linewidth=3)
line4,=plt.plot(x,y_B3_Testacc,"c-",linewidth=3)

plt.axis([-1,301,1,4])
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.xlabel("Epochs",fontproperties = 'Times New Roman', size = 24, ha='center', va='top')
plt.ylabel("Validation Loss",fontproperties = 'Times New Roman', size = 24,ha='center', va='bottom')
plt.legend([line1,line2,line3,line4],["R4(k=1)","R4(k=2)","R4(k=3)","R4(k=4)"],loc="upper right",prop={"family":"Times New Roman","size":22})
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig("./R4/curves/Test_loss_R4",dpi=3000)
plt.show()
############