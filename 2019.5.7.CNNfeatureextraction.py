"""
@author: patrick
1.https://juejin.im/entry/5aea820551882506a36c56d0
https://github.com/ahmedfgad/NumPyCNN/blob/master/NumPyCNN.py
"""
##import------------------------------------------------------------
import sys
import numpy as np
import glob 
#glob可以選取某路徑下的全部檔案
##CNN---------------------------------------------------------------
##define CONVOLUTIONAL LAYER
def conv(raw, conv_filter):
    filter_size = conv_filter.shape[0] #shape[0]指橫列幾列,[1]指直行幾行
    result = np.zeros((raw.shape[0],1)) #make result matrix
    #Looping through the image to apply the convolution operation.
    for r in np.uint16(np.arange(0,raw.shape[0])):#給r:1to最後一列(起始，終點，間隔)(起始，終點)        
        curr_region = raw[r,:] 
            #floor返回不大於輸入參數的最大整數in:0.5,out:0  Ceil返回輸入值的上限整數in:0.5,out:1
            # r-o : r+o , c-o : c+o 當前掃描區域
            #選第r橫列(0是第一列)，選第一直行用(:,0)
            #當前區域與濾波器相乘
        curr_result = curr_region * conv_filter 
        conv_sum = np.sum(curr_result) #Summing the result of multiplication in the matrix.
        result[r,0] = conv_sum #Saving the summation in the convolution layer feature map.只有一個
        
    #Clipping the outliers of the result matrix.
    final_result = result[0:result.shape[0],0] #變成raw.shape[0]列拉，一行
    return final_result

def pooling(final_result,size=2.0,stride=2.0):
    #Preparing the output of the pooling operation.
    #輸出要變成哪個樣子(final_result/2 * 1)
    
    pool_out = np.zeros((int(final_result.shape[0]/size),1))
    #numpy.arange(start,stop,step)
    r2 = 0
    for r in np.arange(0,final_result.shape[0],stride):
        pool_out[r2] = np.max([final_result[r:r+size]])
        r2 = r2 +1 #計算pool_out個數
   # result_pool_out=pool_out[0:]
    return pool_out

def relu(final_result):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(final_result.shape)
    for r in np.arange(0,final_result.shape[0]):
        relu_out[r] = np.max([final_result[r],0]) #([,0])取大於等於0的值
    return relu_out

##open file read file-----------------------------------------------
#filter
    #l1_filter=np.array([-1,0,1])
m=2500     #捲積寬度----------------------------------------------------------------------------改
l1_filter=np.random.randint(-1,1,size=[1,m])  #產生一隨機陣列，randint整數
for i in range(1,241):
    a = np.loadtxt(r'F:\研究所學科\崑山的震動資料\0.52\0.52 (244.0 - 196.9)\t ('+str(i)+').txt')#------注意                    
    input=a.reshape((-1,m))    #轉換成[,b]一列有b個，代表一次計算的時間

    
#------------------------------------------
    print("\n convoluting")
    l1_conv=conv(input,l1_filter)

    print("relu")
    l1_conv_relu=relu(l1_conv)
    print(l1_conv_relu.shape)
    print("pooling")
    l1_conv_relu_pool=pooling(l1_conv_relu,2,2)
    #save
    np.savetxt(r'F:\研究所學科\崑山的震動資料\0.52捲積後\(244-196.9)\T2500('+str(i)+').txt ',l1_conv_relu_pool)#------注意


read_files = glob.glob(r'F:\研究所學科\崑山的震動資料\0.52捲積後\(244-196.9)\*.txt ')

with open(r'F:\研究所學科\崑山的震動資料\0.52捲積後\2500.txt', "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
            
            
    



