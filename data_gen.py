import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
import SCL_flipping as SCLF

size_train = 50000
SNR_in_db = [1.00]


if __name__ == '__main__':

    cross_p = 0.5
    N = 128
    R = 0.5
    M = int(N*R)
    CRC_bits = 16
    List = 4
    T = 20
    O = 1
    Alpha = 1.2
    Batch = 1
    
    polar = SCLF.polar_code(cross_p, N, R, CRC_bits, List, O, T, Alpha)
    
    total_err_bit = np.zeros(len(SNR_in_db))
    total_err_frame = np.zeros(len(SNR_in_db))
    total_ber = np.zeros(len(SNR_in_db))
    total_fer = np.zeros(len(SNR_in_db))
    
    #filename_1 = "SCLF_FER_N"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"
    filename_2 = "X_N"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"
    filename_3 = "Y_N"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"
    filename_4 = "Ysoft_N"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"
    filename_5 = "Flip_N"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"
    filename_6 = "PM"+str(N)+"_CRC"+str(CRC_bits)+"_List"+str(List)+"_Order"+str(O)+"_PW.txt"     

    #f1 = open(filename_1,"w")
    #f2 = open(filename_2,"w")
    #f3 = open(filename_3,"w")
    #f4 = open(filename_4,"w")
    
    for s in range(len(SNR_in_db)):
        snr = SNR_in_db[s]
        Var = 1 / (2 * R * pow(10.0, snr / 10.0))
        sigma = pow(Var, 1 / 2)
        Frame = 0
        
        num_data = 0
        while(num_data <size_train):
            
            X, Y= polar.batch_generator(Batch, sigma)
            for i in range(Batch):
                print(f'SNR ={snr:4.3f}, Frame={Frame}, err={total_err_frame[s]}, num_data={num_data},', end="\r",flush=True)
                Frame += 1
                CRC_flag = 0
                Flip = 0
                Order = 0
                x=X[i]
                y=Y[i]
                
                Y_0, Y_soft, PM, CRC_flag = polar.SCL_decoder(x, Flip, []) # first SCL
                Y_hat = Y_0
                
                locals()['Flip_list'+str(Order+1)] = np.asarray(polar.Flip_choice(PM))
                locals()['Flip_list'+str(Order+1)] = np.expand_dims(locals()['Flip_list'+str(Order+1)], axis=1)
                
                while(np.any(CRC_flag)==False):
                    Flip = 1
                    flip_tmp = [] # used to build the Flip_list of next order
                    if(Order<O):
                        Order += 1
                    else:
                        Y_hat = Y_0
                        #print('flip decoding fail')
                        break
                        
                    j=0
                    while(j<pow(T,Order)):
                        #str_pm = '_'.join([str(x) for x in locals()['Flip_list'+str(Order)][j]])
                        Y_hat, Y_soft, locals()['PM_'+str(Order)], CRC_flag = polar.SCL_decoder(x, Flip, locals()['Flip_list'+str(Order)][j])
                        if((np.any(CRC_flag)==True)):
                            flip_correct = locals()['Flip_list'+str(Order)][j]
                            with open(filename_2, 'a') as f2:
                                for value in x:
                                    f2.write(f'{value} ')
                                f2.write('\n')
                            with open(filename_3, 'a') as f3:
                                for value in y:
                                    f3.write(f'{value} ')
                                f3.write('\n')
                            with open(filename_4, 'a') as f4:
                                for value in Y_soft:
                                    f4.write(f'{value} ')
                                f4.write('\n')
                            with open(filename_5, 'a') as f5:
                                    for value in flip_correct:
                                        f5.write(f'{value} ')
                                    f5.write('\n')
                            with open(filename_6, 'a') as f6:
                                    for value in PM:
                                        f6.write(f'{value} ')
                                    f5.write('\n')        
                            num_data += 1
                            #print('flip decoding succeed')
                            break
                            
                        Idx_Flip = polar.Flip_choice(locals()['PM_'+str(Order)])
                        #print(j,' ',len(Idx_Flip),end='\n')
                        flip_tmp.append(Idx_Flip)
                        j=j+1
                        
                    if((np.any(CRC_flag)==False)):
                        flip_n = np.expand_dims((np.asarray(flip_tmp)).flatten(), axis=1)
                        #print(flip_n.shape)
                        locals()['Flip_list'+str(Order+1)] = np.repeat(locals()['Flip_list'+str(Order)], T, axis=0)
                        #print(locals()['Flip_list'+str(Order+1)].shape)
                        locals()['Flip_list'+str(Order+1)] = np.append(locals()['Flip_list'+str(Order+1)], flip_n, axis=1)
                        #print(locals()['Flip_list'+str(Order+1)].shape)
                        
                err_bit = np.sum(np.not_equal(Y_hat, Y[i]))
                arr_err = np.not_equal(Y_hat, Y[i])
                err_frame = np.sum((np.sum(arr_err)).astype(bool, copy=False))
                #print(err_frame)
                total_err_bit[s] += err_bit
                total_err_frame[s] += err_frame
            
        #print('\n')
        total_ber[s] = total_err_bit[s]/(Frame*M)
        total_fer[s] = total_err_frame[s]/Frame
        
    print(total_fer)
    #f1 = open(filename_1,"w")
    #f1.write(str(total_fer)+"\n")
    