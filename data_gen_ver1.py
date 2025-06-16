import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
import SCLF as SCLF

#size_train = 50000
size_train = 50 #testing
SNR_in_db = [1.00]

cross_p = 0.5
N = 1024
R = 0.5
M = int(N*R)
CRC_bits = 0
List = 1
T = 20
O = 1
Alpha = 1.2


if __name__ == '__main__':
    
    polar = SCLF.polar_code(cross_p, N, R, CRC_bits, List, O, T, Alpha)
    
    total_err_bit = np.zeros(len(SNR_in_db))
    total_err_frame = np.zeros(len(SNR_in_db))
    total_ber = np.zeros(len(SNR_in_db))
    total_fer = np.zeros(len(SNR_in_db))
    
    llr_all=[] #(num_samples, N)
    y_all=[]#(num_samples, N)
    y_soft_all=[]#(num_samples, N)
    pm_all=[]#(num_samples, N, 2L)
    crc_sydrome_all=[]#(num_samples, syndrome_len) equals (num_samples, r)
    flip_arr_all=[]#(num_samples, N)
    flip_bool_all=[]#(num_samples, 1) determine whether the induced noise can be fixed or not
    
    for s in range(len(SNR_in_db)):
        snr = SNR_in_db[s]
        Var = 1 / (2 * R * pow(10.0, snr / 10.0))
        sigma = pow(Var, 1 / 2)
        Frame = 0
        
        num_data = 0
        while(num_data <size_train):
            
            LLR, Y= polar.batch_generator(sigma)
        
            print(f'SNR ={snr:4.3f}, Frame={Frame}, err={total_err_frame[s]}, num_data={num_data},', end="\r",flush=True)
            Frame += 1
            Flip_func = 0
            Order = 0
            
            Y_hat, PM = polar.scl_decoder(LLR, []) # first SCL with null flip array and Flip_func set to 0
            #Y_hat, Y_soft, PM, CRC_syndrome = SCLF.scl_decoder(LLR, polar.info_set, List, []) # first SCL with null flip array and Flip_func set to 0
            Y_hat_0 = Y_hat.copy()
            #Y_soft_0 = Y_soft.copy()
            PM_0 = PM.copy()                
            #CRC_syndrome_0 = CRC_syndrome.copy()
            '''
            # 初始化字典以儲存 Flip_list
            Flip_lists = {}
            # 產生 Flip_list[Order + 1]
            Flip_lists[Order + 1] = np.asarray(polar.Flip_choice(PM))
            Flip_lists[Order + 1] = np.expand_dims(Flip_lists[Order + 1], axis=1)
            
            while(np.any(CRC_syndrome)==1): #syndrome are nonzero
                Flip_func = 1
                flip_tmp = [] # used to build the Flip_list of next order
                if(Order<O):
                    Order += 1
                else:
                    Y_hat = Y_hat_0
                    ##print('flip decoding fail')
                    #collect failure flipping result
                    
                    ##flip_arr = np.zeros(N, dtype=np.uint8)
                    ##flip_arr[:] = -1                #set to -1 means we doesn't find out correct flip arr 
                    ##llr_all.append(LLR)
                    ##y_all.append(Y_hat_0)
                    ##y_soft_all.append(Y_soft_0)
                    ##pm_all.append(PM_0)
                    ##crc_sydrome_all.append(CRC_syndrome_0)
                    ##flip_arr_all.append(flip_arr)
                    ##flip_bool_all.append(0)
                    ##
                    ##num_data += 1
                    
                    break 
                    
                j=0
                while(j<pow(T,Order)):
                    
                    Y_hat, Y_soft, PM, CRC_syndrome = polar.scl_decoder(LLR, Flip_lists[Order][j])
                    if(np.any(CRC_syndrome)==False): #syndrome are all zeros                    
                        #collect successful flip result
                        flip_arr= np.zeros(N, dtype=np.uint8)
                        flip_arr[(Flip_lists[Order][j])] = 1    #Flip_lists[Order][j] is the correct flip index
                                                                #convert index to True False array for Machine learning
                        llr_all.append(LLR)
                        y_all.append(Y_hat)
                        y_soft_all.append(Y_soft)
                        pm_all.append(PM)
                        crc_sydrome_all.append(CRC_syndrome)
                        flip_arr_all.append(flip_arr)
                        flip_bool_all.append(1)
                        
                        num_data += 1
                        ##print('flip decoding succeed')
                        break
                        
                    Idx_Flip = polar.Flip_choice(PM) #list(T,)
                    flip_tmp.append(Idx_Flip) #flip_tmp(T,T)
                    j=j+1
                    
                if(np.any(CRC_syndrome) == True):

                    # convert flip_tmp from T times (T,) list to (T^2,1) array
                    flip_n = np.expand_dims(np.concatenate(flip_tmp), axis=1)
                    ##flip_n = np.expand_dims(np.asarray(flip_tmp).flatten(), axis=1) #another option for data processing
                    # 重複 Flip_list[Order] 並附加 flip_n
                    Flip_lists[Order+1] = np.repeat(Flip_lists[Order], T, axis=0)
                    Flip_lists[Order+1] = np.append(Flip_lists[Order+1], flip_n, axis=1)
            '''
            err_bit = np.sum(Y_hat != Y)
            err_frame = int(np.any(Y_hat != Y))
            #print(err_frame)
            total_err_bit[s] += err_bit
            total_err_frame[s] += err_frame
        
        #save the collected data
        ##np.savetxt('llr_all.txt', llr_all, fmt='%.8f')
        ##np.savetxt('y_all.txt', y_all, fmt='%.8f')
        ##np.savetxt('y_soft_all.txt', y_soft_all, fmt='%.8f')
        ##np.savetxt('pm_all.txt', pm_all, fmt='%.8f')
        ##np.savetxt('flip_arr_all.txt', flip_arr_all, fmt='%.8f')
        ##np.savetxt('flip_bool_all.txt', flip_bool_all,)
        
        total_ber[s] = total_err_bit[s]/(Frame*M)
        total_fer[s] = total_err_frame[s]/Frame
        
    print(total_fer)
