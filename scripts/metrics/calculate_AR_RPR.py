import pandas as pd
import numpy as np
from statistics import mean
from os import path as osp


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def cal_AR_RPR(ExcellenceLine, AcceptanceLine, realSRModel):
    RPR_acceptance = []
    RPR_unacceptance = []
    RPR_list=[]

    Positive_value = 0
    for i in range(len(realSRModel)):
        RPR = sigmoid((realSRModel[i] - AcceptanceLine[i])/(ExcellenceLine[i] - AcceptanceLine[i]))
        RPR_list.append(RPR)

        if realSRModel[i] >= AcceptanceLine[i]:
            Positive_value +=1 
            RPR_acceptance.append(RPR)
        else:
            RPR_unacceptance.append(RPR)

    AR = round((Positive_value/len(realSRModel)), 2)
    m_RPR = round(mean(RPR_list), 2)

    if len(RPR_acceptance):
        mean_RPR_acceptance = np.round(mean(RPR_acceptance), 2)
    else:
        mean_RPR_acceptance = 0.00

    if len(RPR_unacceptance):
        mean_RPR_unacceptance = np.round(mean(RPR_unacceptance), 2)
    else:
        mean_RPR_unacceptance = 0.00
        
    return AR, m_RPR, mean_RPR_acceptance, mean_RPR_unacceptance, RPR_list



def calculate_seal(realSRModel, line_data, iqa):

    AcceptanceLine= line_data['fsrcnn'+'_'+iqa]
    ExcellenceLine = line_data['srresnet'+'_'+iqa]

 ############################# pop value #############################
    if iqa == 'PSNR':
        for i in range(len(realSRModel)):
            if ExcellenceLine[i] - AcceptanceLine[i] < 0.1:
                AcceptanceLine.pop(i)
                ExcellenceLine.pop(i)
                realSRModel.pop(i)
    elif iqa == 'SSIM':
        for i in range(len(realSRModel)):
            if ExcellenceLine[i] - AcceptanceLine[i] < 0.01:
                AcceptanceLine.pop(i)
                ExcellenceLine.pop(i)
                realSRModel.pop(i)

    # elif iqa == 'LPIPS':
    #     for i in range(len(realSRModel)):
    #         if ExcellenceLine[i] - AcceptanceLine[i] < 0.01:
    #             AcceptanceLine.pop(i)
    #             ExcellenceLine.pop(i)
    #             realSRModel.pop(i)
 
    ExcellenceLine = ExcellenceLine.tolist()
    AcceptanceLine = AcceptanceLine.tolist()
    realSRModel = realSRModel.tolist()
 ##################################################################


    AR, mRPR, RPRA, RPRU, RPR_PSNR = cal_AR_RPR(ExcellenceLine, AcceptanceLine, realSRModel)

    data_RPR = {}
    data_RPR['RPR_PSNR'] = RPR_PSNR 
    data_rpr1 = pd.DataFrame(data_RPR)
    rpr_quant1 = data_rpr1.quantile([0, .25, .5, .75, 1], axis = 0)
    quantile_list1 = rpr_quant1['RPR_PSNR'].tolist()
    IQR = quantile_list1[3] - quantile_list1[1]
    IQR = round(IQR,2)

    return AR, IQR, RPRA, RPRU

def calculate_list(data, line_data, iqa, model_name_list, csv_rootpath):
    data_iqa={}
    data_iqa['metric'] = ['AR','IQR','RPRA','RPRU']
    for i in range(len(model_name_list)):
        data_key = model_name_list[i]+'_'+iqa
        results_list = data[data_key]
        AR,IQR,RPRA,RPRU = calculate_seal(results_list, line_data, iqa)
        data_iqa[model_name_list[i]] = [AR,IQR,RPRA,RPRU]
    data_iqa = pd.DataFrame(data_iqa)
    data_iqa.to_csv(osp.join(csv_rootpath,f'AR_RPR_on_{iqa}.csv'))


if __name__ == '__main__':

    model_csv_path = 'scripts/metrics/Set14_SE/model.csv'
    line_csv_path = 'scripts/metrics/Set14_SE/line.csv'

    csv_save_rootpath = 'scripts/metrics/Set14_SE'

    model_name_list = ['SRResNet','DASR','BSRNet', 'RealESRNet_x4plus', 'Real-RRDB_withdropout', 'GDRealESRNetx4plus' ,'swinir-l']

    data = pd.read_csv(model_csv_path)
    line_data = pd.read_csv(line_csv_path)

    calculate_list(data,line_data, 'psnr', model_name_list, csv_save_rootpath)
    calculate_list(data,line_data, 'ssim', model_name_list, csv_save_rootpath)
    # calculate_list(data,line_data, 'lpips', model_name_list, csv_save_rootpath)
    # calculate_list(data,line_data, 'niqe', model_name_list, csv_save_rootpath)